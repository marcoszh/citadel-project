#!/bin/bash
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

Copyright (C) 2017-2018 scontain.com
'

#
# - install patched sgx driver 


set -e

# print the right color for each level
#
# Arguments:
# 1:  level

function msg_color {
    priority=$1
    if [[ $priority == "fatal" ]] ; then
        echo -e "\033[31m"
    elif [[ $priority == "error" ]] ; then
        echo -e "\033[34m"
    elif [[ $priority == "warning" ]] ; then
        echo -e "\033[35m"
    elif [[ $priority == "info" ]] ; then
        echo -e "\033[36m"
    elif [[ $priority == "debug" ]] ; then
        echo -e "\033[37m"
    elif [[ $priority == "default" ]] ; then
        echo -e "\033[00m"
    else
        echo -e "\033[32m";
    fi
}

function no_error_message {
    exit $? 
}

function issue_error_exit_message {
    errcode=$?
    trap no_error_message EXIT
    if [[ $errcode != 0 ]] ; then
        msg_color "fatal"
        echo -e "ERROR: installation of SGX driver failed (script=install_sgx_driver.sh, Line: ${BASH_LINENO[0]}, ${BASH_LINENO[1]})"
        msg_color "default"
    else
        msg_color "OK"
        echo "OK"
        msg_color "default"
    fi
    exit $errcode
}
trap issue_error_exit_message EXIT

function verbose
{
    echo $@
}

function log_error
{
    msg_color "error"
    echo $@
    msg_color "default"
    exit 1
}

function check_driver {
    if [[ ! -e /sys/module/isgx/version ]] ; then
        install_driver=true
    else 
        install_driver=false
        verbose "SGX-driver already installed."
        if [[ ! -e /dev/isgx ]] ; then
            log_error "SGX driver is installed but no SGX device - SGX not enabled?"
        fi
    fi
}


function install_sgx_driver {
    if [[ $install_driver == true ]] ; then
        dir=$(mktemp -d)
        cd "$dir"
        rm -rf linux-sgx-driver
        git clone https://github.com/fortanix/linux-sgx-driver 

        cd linux-sgx-driver/
        sudo apt-get update
        sudo apt-get install -y build-essential
        sudo apt-get install -y linux-headers-$(uname -r)
        make 

        sudo mkdir -p "/lib/modules/"`uname -r`"/kernel/drivers/intel/sgx"    
        sudo cp -f isgx.ko "/lib/modules/"`uname -r`"/kernel/drivers/intel/sgx"    

        sudo sh -c "cat /etc/modules | grep -Fxq isgx || echo isgx >> /etc/modules"    
        sudo /sbin/depmod
        sudo /sbin/modprobe isgx

        cd ..
    fi
}

check_driver
install_sgx_driver