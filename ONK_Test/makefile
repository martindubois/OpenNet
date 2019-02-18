
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       ONK_Test/makefile

CFLAGS = -g

INCLUDES = -I ../Includes -I ../Import/Includes

LIBRARIES = ../Import/Libraries/KmsLib.a

OUTPUT = ../Binaries/ONK_Test

SOURCES =	Device.cpp   \
			ONK_Test.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(CFLAGS) -o $@ $(INCLUDES) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets / Cibles ====================================================

$(OUTPUT) : $(OBJECTS) $(LIBRARIES)
	g++ -pthread -o $@ $^

depend:
	makedepend -Y $(INCLUDES) $(SOURCES)

# DO NOT DELETE

Device.o: ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Device.o: ../Import/Includes/WindowsToLinux.h
Device.o: ../Import/Includes/KmsLib/DriverHandle.h
Device.o: ../Import/Includes/KmsLib/FileHandle.h
Device.o: ../Import/Includes/KmsLib/Exception.h
Device.o: ../Import/Includes/KmsLib/ThreadBase.h ../Import/Includes/KmsTest.h
Device.o: ../Includes/OpenNetK/Adapter_Types.h
Device.o: ../Includes/OpenNetK/PacketGenerator_Types.h ../Common/Constants.h
Device.o: ../Common/IoCtl.h
ONK_Test.o: ../Import/Includes/KmsTest.h ../Import/Includes/KmsBase.h
ONK_Test.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
