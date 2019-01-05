
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

