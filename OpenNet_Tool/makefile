
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       OpenNet_Tool/makefile

CFLAGS = -g

INCLUDES = -I ../Includes -I ../Import/Includes

LIBRARIES = ../Binaries/OpenNet.so ../Libraries/TestLib.a ../Import/Libraries/KmsLib.a

OUTPUT = ../Binaries/OpenNet_Tool

SOURCES =	OpenNet_Tool.cpp

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

