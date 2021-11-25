#!//bin/bash
: '
Access to this file is granted under the SCONE COMMERCIAL LICENSE V1.0

Any use of this product using this file requires a commercial license from scontain UG, www.scontain.com.

Permission is also granted  to use the Program for a reasonably limited period of time  (but no longer than 1 month)
for the purpose of evaluating its usefulness for a particular purpose.

THERE IS NO WARRANTY FOR THIS PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING
THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE,
YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED ON IN WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY
MODIFY AND/OR REDISTRIBUTE THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL,
INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM INCLUDING BUT NOT LIMITED TO LOSS
OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE
WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Copyright (C) 2018 scontain.com
'

set -e -x


RED='\033[0;31m'
NC='\033[0m' # No Color
GREEN='\033[0;32m'
ver=true

errorexit() {
  printf "${RED}#####  An error occurred while installing host! Please check the logs. Sometimes it is sufficient to restart this script! #####${NC}\n"
  exit 1
}

trap 'errorexit' ERR


function verbose {
    if [[ $ver != "" ]]  ; then
        printf "${GREEN}$1${NC}\n"
    fi
}

verbose "..downloading more files"

curl -fssl https://raw.githubusercontent.com/scontain/install_dependencies/master/las.service --output /tmp/las.service
curl -fssl https://raw.githubusercontent.com/scontain/install_dependencies/master/las-docker-compose.yml --output /tmp/las-docker-compose.yml
curl -fssl https://raw.githubusercontent.com/scontain/install_dependencies/master/microcode-load.service --output /tmp/microcode-load.service
curl -fssl https://raw.githubusercontent.com/scontain/install_dependencies/master/docker --output /tmp/docker

verbose "..installing microcode update"

installed=$(systemctl status microcode-load.service | grep "Started updated microcode-load service." | wc -l)
if [[ $installed == "1" ]] ; then
    verbose "  microcode update already installed - skipping"
else
    TMPDIR=$(mktemp -d)
    cd $TMPDIR
    git clone https://github.com/intel/Intel-Linux-Processor-Microcode-Data-Files.git
    cd Intel-Linux-Processor-Microcode-Data-Files
    sudo apt-get update
    sudo apt-get install -y intel-microcode
    if [ -f /sys/devices/system/cpu/microcode/reload ] ; then
        if [ -d /lib/firmware ] ; then
            mkdir -p OLD
            cp -rf /lib/firmware/intel-ucode OLD
                sudo cp -rf intel-ucode /lib/firmware
            echo "1" | sudo tee /sys/devices/system/cpu/microcode/reload
        else
            echo "Error: microcode directory does not exist"
        fi

        verbose "..enable start of new microcode on each reboot"

        cat > /tmp/load-intel-ucode.sh << EOF
#!/bin/bash
echo "1" | sudo tee /sys/devices/system/cpu/microcode/reload
EOF
        sudo mv -f /tmp/load-intel-ucode.sh /lib/firmware/load-intel-ucode.sh
        chmod a+x /lib/firmware/load-intel-ucode.sh

        sudo mv -f /tmp/microcode-load.service  /etc/systemd/system/microcode-load.service
        sudo systemctl daemon-reload
        sudo systemctl start microcode-load.service
        sudo systemctl enable microcode-load.service || echo "looks like microcode-load.service  is already enabled"
    else
        echo "Error: check that package intel-microcode is installed"
    fi
fi

verbose "..installing  docker engine"

verbose "..installing docker"
installed=$(which docker | wc -l)
if [[ $installed == "1" ]] ; then
    verbose "  docker  already installed - skipping"
else
    curl -fssl https://raw.githubusercontent.com/SconeDocs/SH/master/install_docker.sh | bash
fi

if [[ ! -f /usr/local/bin/docker ]]; then
    sudo cp -f /tmp/docker /usr/local/bin/docker
    sudo chmod +x /usr/local/bin/docker
fi

verbose "..installing docker compose"
installed=$(which docker-compose | wc -l)
if [[ $installed == "1" ]] ; then
    verbose "  docker-compose already installed - skipping"
else
    sudo curl -L "https://github.com/docker/compose/releases/download/1.22.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    # simple check
    docker-compose --version
fi

verbose "..installing SGX driver"

if [[ -c /dev/isgx ]] || [[ -c /dev/sgx ]] ; then
    verbose "  /dev/(i)sgx already installed - skipping"
else
    curl -fssl https://raw.githubusercontent.com/SconeDocs/SH/master/install_sgx_driver.sh | bash
fi

verbose "..ensure that we can run docker without sudo"

if getent group ubuntu | grep &>/dev/null "\bubuntu\b"; then
    verbose "  user and group ubuntu already exist"
else
    sudo groupadd ubuntu || verbose "  group ubuntu already exist"
    sudo adduser --ingroup ubuntu ubuntu || verbose "  user ubuntu already exists!"
fi

if getent group docker | grep &>/dev/null "\bubuntu\b"; then
    todo=false
else
    todo=true
fi
if [[ $todo == true ]] ; then
    sudo groupadd docker || verbose "  group docker already exist"
    sudo gpasswd -a ubuntu docker || verbose "  ubuntu is already member of group docker"
fi

verbose "..installing LAS service"

installed=$(systemctl status las.service | grep "Active: active (running)" | wc -l)

if [[ "$1" == "-f" ]] ; then
    if [[ $installed -gt 0 ]] ; then
        verbose "  force flag given: stopping running las service"
        sudo systemctl stop las.service || verbose "failed to stop las service ... continue anyhow"
        sudo systemctl disable las.service || verbose "failed to disable las service ... continue anyhow"
        installed=0
    fi
fi

if [[ $installed -gt 0 ]] ; then
    verbose "  LAS service already installed - skipping"
else
    if [[ -c /dev/isgx ]]; then
      SGX_DEVICE="/dev/isgx"
    elif [[ -c /dev/sgx ]]; then
      SGX_DEVICE="/dev/sgx"
    fi
    sudo mkdir -p /home/ubuntu/las
    SGX_DEVICE=$SGX_DEVICE envsubst < /tmp/las-docker-compose.yml > /tmp/docker-compose-las.yml
    sudo mv /tmp/docker-compose-las.yml /home/ubuntu/las/docker-compose.yml
    sudo rm -f /tmp/las-docker-compose.yml

    #export DOCKER_CONTENT_TRUST=1

    sudo mv -f /tmp/las.service  /etc/systemd/system/las.service
    sudo docker pull sconecuratedimages/iexecsgx:las
    sudo systemctl daemon-reload
    sleep 2
    sudo systemctl start las.service
    sudo systemctl status las.service
    sudo systemctl enable las.service || echo "looks like las.service  is already enabled"
fi