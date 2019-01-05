
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet_Test/makefile

CFLAGS = -g

INCLUDES = -I ../Includes -I ../Import/Includes

LIBRARIES = ../Binaries/OpenNet.so ../Import/Libraries/KmsLib.a

OUTPUT = ../Binaries/OpenNet_Test

SOURCES =	Adapter.cpp         \
			Base.cpp            \
			BlackHole.cpp       \
			EthernetAddress.cpp \
			Kernel.cpp          \
			Kernel_Forward.cpp  \
			Mirror.cpp          \
			Processor.cpp		\
			SourceCode.cpp		\
			Status.cpp          \
			System.cpp			\
			OpenNet_Test.cpp    \
            SetupA.cpp

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

