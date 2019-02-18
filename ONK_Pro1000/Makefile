
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       ONK_Pro1000/makefile

-include ../KernelCppFlags.mk

EXTRA_CFLAGS := -I$(src)/../Includes -I/home/mdubois/NVIDIA-Linux-x86_64-410.93/kernel/nvidia

obj-m += ONK_Pro1000.o

ONK_Pro1000-objs := Device_Linux.o DeviceCpp.o Driver_Linux.o NvBuffer.o Pro1000.o ../Libraries/ONK_Lib.a
list-multi := ONK_Pro1000.o

# ===== Rules ===============================================================

.cpp.o:
	g++ -c -I ../Includes -o $@ $(KERNEL_CPP_FLAGS) $<

# ===== Targets =============================================================

all: DeviceCpp.o Pro1000.o module

module:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

# ===== C++ files ===========================================================

DeviceCpp.o: DeviceCpp.cpp

Pro1000.o: Pro1000.cpp
