
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2019 KMS. All rights reserved.
# Product    OpenNet
# File       ONK_Tunnel_IO/makefile

-include ../Kernel.mk

EXTRA_CFLAGS := -I$(src)/../Includes -I/home/mdubois/NVIDIA-Linux-x86_64-410.93/kernel/nvidia

obj-m += ONK_Tunnel_IO.o

ONK_Tunnel_IO-objs := Device_Linux.o DeviceCpp.o Driver_Linux.o VirtualHardware.o ../Libraries/ONK_Lib.a
list-multi := ONK_Tunnel_IO.o

# ===== Rules ===============================================================

.cpp.o:
	g++ -c -I ../Includes -o $@ $(KERNEL_CPP_FLAGS) $<

# ===== Targets =============================================================

all: DeviceCpp.o VirtualHardware.o module

module:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

# ===== C++ files ===========================================================

DeviceCpp.o: DeviceCpp.cpp
