
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All right reserved.
# Product    OpenNet
# File       KernelCppFlags.mk

KERNEL_CPP_FLAGS = -D_KMS_LINUX_ \
	-ffreestanding -fno-asynchronous-unwind-tables -fno-builtin -fno-exceptions -fno-pic -fno-stack-check -fno-rtti -funit-at-a-time \
	-m64 -mcmodel=kernel \
	-nostdinc \
	-O2
