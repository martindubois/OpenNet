
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
			SpinLock.cpp       \
			SpinLock_Linux.cpp

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
	
