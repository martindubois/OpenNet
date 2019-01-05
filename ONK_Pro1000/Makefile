
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       ONK_Pro1000/makefile

obj-m += ONK_Pro1000.o

ONK_Pro1000-objs := Device_Linux.o DeviceCpp.o Driver_Linux.o Pro1000.o ../Libraries/ONK_Lib.a
list-multi := ONK_Pro1000.o

# ===== Rules ===============================================================

.cpp.o:
	g++ -c -D_KMS_LINUX_ -I ../Includes -o $@ -fno-builtin -fno-exceptions -fno-pic -fno-rtti -nostdinc $<

# ===== Targets =============================================================

all: DeviceCpp.o Pro1000.o module

module:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

# ===== C++ files ===========================================================

DeviceCpp.o: DeviceCpp.cpp

Pro1000.o: Pro1000.cpp
