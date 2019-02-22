
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All right reserved.
# Product    OpenNet
# File       ONK_Lib/makefile

include ../KernelCppFlags.mk

INCLUDES = -I ../Includes

OUTPUT = ../Libraries/ONK_Lib.a

SOURCES =	Adapter.cpp        \
			Adapter_Linux.cpp  \
			Hardware.cpp       \
			Hardware_Linux.cpp \
			Linux.cpp          \
			Packet.cpp         \
			SpinLock.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(CFLAGS) -o $@ $(INCLUDES) $(KERNEL_CPP_FLAGS) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets ==============================================================

$(OUTPUT) : $(OBJECTS)
	ar r $@ $^
	ranlib $@

depend:
	makedepend -Y $(INCLUDES) $(SOURCES)
	
# DO NOT DELETE

Adapter.o: Component.h ../Includes/OpenNetK/OS.h ../Includes/OpenNetK/Debug.h
Adapter.o: ../Includes/OpenNetK/StdInt.h ../Includes/OpenNetK/Hardware.h
Adapter.o: ../Includes/OpenNetK/Adapter_Types.h ../Includes/OpenNetK/Types.h
Adapter.o: ../Includes/OpenNetK/Packet.h ../Includes/OpenNetK/SpinLock.h
Adapter.o: ../Includes/OpenNetK/OSDep.h ../Includes/OpenNetK/Adapter.h
Adapter.o: ../Includes/OpenNetK/Constants.h ../Includes/OpenNetK/IoCtl.h
Adapter.o: ../Includes/OpenNetK/PacketGenerator_Types.h ../Common/Constants.h
Adapter.o: ../Common/IoCtl.h ../Common/OpenNetK/Adapter_Statistics.h
Adapter.o: ../Common/Version.h
Adapter_Linux.o: ../Includes/OpenNetK/OS.h ../Includes/OpenNetK/StdInt.h
Adapter_Linux.o: ../Includes/OpenNetK/Adapter.h
Adapter_Linux.o: ../Includes/OpenNetK/Adapter_Types.h
Adapter_Linux.o: ../Includes/OpenNetK/Constants.h
Adapter_Linux.o: ../Includes/OpenNetK/IoCtl.h
Adapter_Linux.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Adapter_Linux.o: ../Includes/OpenNetK/Types.h ../Includes/OpenNetK/OSDep.h
Adapter_Linux.o: ../Includes/OpenNetK/Hardware_Linux.h
Adapter_Linux.o: ../Includes/OpenNetK/Adapter_Linux.h ../Common/Constants.h
Adapter_Linux.o: ../Common/IoCtl.h
Hardware.o: ../Includes/OpenNetK/OS.h ../Includes/OpenNetK/StdInt.h
Hardware.o: ../Includes/OpenNetK/Adapter.h
Hardware.o: ../Includes/OpenNetK/Adapter_Types.h
Hardware.o: ../Includes/OpenNetK/Constants.h ../Includes/OpenNetK/IoCtl.h
Hardware.o: ../Includes/OpenNetK/PacketGenerator_Types.h
Hardware.o: ../Includes/OpenNetK/Types.h ../Includes/OpenNetK/OSDep.h
Hardware.o: ../Includes/OpenNetK/Hardware_Statistics.h
Hardware.o: ../Includes/OpenNetK/SpinLock.h ../Includes/OpenNetK/Hardware.h
Hardware.o: ../Common/Constants.h ../Common/Version.h
Hardware_Linux.o: ../Includes/OpenNetK/OS.h ../Includes/OpenNetK/StdInt.h
Hardware_Linux.o: ../Includes/OpenNetK/Hardware.h
Hardware_Linux.o: ../Includes/OpenNetK/Adapter_Types.h
Hardware_Linux.o: ../Includes/OpenNetK/Types.h
Hardware_Linux.o: ../Includes/OpenNetK/Hardware_Linux.h
Packet.o: ../Includes/OpenNetK/OS.h ../Includes/OpenNetK/StdInt.h
Packet.o: ../Includes/OpenNetK/Types.h ../Includes/OpenNetK/Packet.h
SpinLock.o: ../Includes/OpenNetK/OS.h ../Includes/OpenNetK/StdInt.h
SpinLock.o: ../Includes/OpenNetK/SpinLock.h ../Includes/OpenNetK/OSDep.h
SpinLock_Linux.o: ../Includes/OpenNetK/OS.h ../Includes/OpenNetK/StdInt.h
