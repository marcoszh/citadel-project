import math
import copy
import threading

MAX_ITER = 100

class Processor:
    def __init__(self, input_size,
                    aggr_func,
                    level, num,
                    decrypt_func, encrypt_func, 
                    export_func, logger):
        self.input_size = input_size
        self.aggr_func = aggr_func
        self.level = level
        self.num = num
        self.decrypt_func = decrypt_func
        self.encrypt_func = encrypt_func
        self.export_func = export_func
        self.logger = logger
        self.locks = []
        self.result = dict()
        self.input_count = dict()
        for i in range(MAX_ITER):
            self.input_count[i] = 0
            lock = threading.Lock()
            self.locks.append(lock)

    def feed_data(self, iter_num, data):
        # if input_size = 1, then no need to decrypt & encrypt
        self.locks[iter_num].acquire()
        self.input_count[iter_num] += 1
        self.logger.info("[processor/level {}/num {}] receive data. input_count = {}, required_size = {}".
                            format(self.level, self.num, self.input_count[iter_num], self.input_size))

        if self.input_size > 1:
            addi = self.decrypt_func(data)

            if iter_num not in self.result:
                self.result[iter_num] = addi
            else:
                self.result[iter_num] = self.aggr_func(self.result[iter_num], addi)

            self.logger.info("[processor/level {}/num {}] finish process data. input_count = {}, required_size = {}".
                            format(self.level, self.num, self.input_count[iter_num], self.input_size))            

            if self.input_count[iter_num] == self.input_size:
                self.locks[iter_num].release()
                self.logger.info("[processor/level {}/num {}] lock released".
                        format(self.level, self.num))
                        
                out_data = self.encrypt_func(self.result[iter_num])
                self.logger.info("[processor/level {}/num {}] aggregation finished".
                                    format(self.level, self.num))
                self.export_func({
                    "iter_num": iter_num,
                    "data": out_data,
                    "level": self.level,
                    "num": self.num
                })
            else:
                self.locks[iter_num].release()
                self.logger.info("[processor/level {}/num {}] lock released".
                        format(self.level, self.num))

        else:
            self.locks[iter_num].release()
            self.logger.info("[processor/level {}/num {}] lock released".
                    format(self.level, self.num))

            # input_size = 1, directly send out this data
            self.logger.info("[processor/level {}/num {}] aggregation finished".
                    format(self.level, self.num))

            self.export_func({
                "iter_num": iter_num,
                "data": data,
                "level": self.level,
                "num": self.num
            })

class AggrTreeProcessorCenter:
    def __init__(self, worker_num, batch_size, this_worker_id,
                    encrypt_func, decrypt_func, aggr_func,
                    pushup_func, next_iter_func, ip_list,
                    logger):
        self.worker_num = worker_num
        self.batch_size = batch_size
        self.this_worker_id = this_worker_id
        self.pushup_func = pushup_func
        self.next_iter_func = next_iter_func
        self.encrypt_func = encrypt_func
        self.decrypt_func = decrypt_func
        self.aggr_func = aggr_func
        self.ip_list = ip_list
        self.logger = logger
        self.processors = dict()
        self.prepare_args()

    def prepare_args(self):
        # tree height
        self.height = 1
        while math.pow(self.batch_size, self.height - 1) < self.worker_num:
            self.height += 1
        # size of each level of the tree
        self.level_size = [0] * self.height
        self.level_size[self.height - 1] = self.worker_num
        for i in range(self.height - 2, -1, -1):
            self.level_size[i] = math.ceil(self.level_size[i + 1] / self.batch_size)
        # worker_id
        self.worker_id = dict()
        for i in range(self.height):
            counter = 0
            for j in range(i + 1, self.height):
                counter += self.level_size[j]
            counter %= self.worker_num
            for j in range(self.level_size[i]):
                self.worker_id[(i + 1, j)] = counter
                if counter == self.this_worker_id:
                    self.processors[(i + 1, j)] = self.build_processor(i + 1, j)
                counter = (counter + 1) % self.worker_num

    def build_processor(self, level, num):
        input_size = self.get_input_size(level, num)
        
        if level == 1:
            processor = Processor(
                input_size = input_size,
                aggr_func = self.aggr_func,
                level = level,
                num = num,
                encrypt_func = self.encrypt_func,
                decrypt_func = self.decrypt_func,
                export_func = self.next_iter_func,
                logger = self.logger
            )
            return processor

        destination_worker = self.get_destination_worker_id(level, num)
        dest_ip = self.ip_list[destination_worker]
        this_pushup_func = lambda payloads: self.pushup_func({
            **payloads,
            "ip": dest_ip
        })
        processor = Processor(
            input_size = input_size,
            aggr_func = self.aggr_func,
            level = level,
            num = num,
            encrypt_func = self.encrypt_func,
            decrypt_func = self.decrypt_func,
            export_func = this_pushup_func,
            logger = self.logger
        )

        return processor
        
    
    def get_input_size(self, level, num):
        if level == self.height:
            return 1
        if num < self.get_level_size(level) - 1:
            return self.batch_size
        else:
            total = self.get_level_size(level + 1)
            return total - self.batch_size * (self.get_level_size(level) - 1)

    def get_height(self):
        return self.height

    def get_level_size(self, level):
        return self.level_size[level - 1]

    def get_worker_id(self, level, num):
        return self.worker_id[(level, num)]

    def get_destination_worker_id(self, level, num):
        return self.worker_id[(level - 1, math.floor(num / self.batch_size))]

    def get_destination_tree_node(self, level, num):
        if level == -1:
            return (self.height, num)
        return (level - 1, math.floor(num / self.batch_size))

    def set_this_worker_id(self, worker_id):
        self.this_worker_id = worker_id
    
    def set_ip_list(self, ip_list):
        self.ip_list = ip_list

    def receive_data(self, iter_num, data, level_from, num_from):
        to_node = self.get_destination_tree_node(level_from, num_from)
        self.logger.info("to node {}".format(to_node))
        self.processors[to_node].feed_data(iter_num, data)

if __name__ == '__main__':
    aggr = AggrTreeProcessorCenter(
        worker_num = 15,
        batch_size = 2, 
        this_worker_id = 14,
        encrypt_func = lambda x: x,
        decrypt_func = lambda x: x,
        aggr_func = lambda x, y: x + y,
        pushup_func = lambda params: print("pushup: {}".format(params)),
        next_iter_func = lambda params: print("next_iter: {}".format(params)),
        ip_list = ["http://{}".format(x) for x in range(30)]
    )

