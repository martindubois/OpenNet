
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       TestLib/makefile

CFLAG = -g

INCLUDES = -I ../Includes -I ../Import/Includes

OUTPUT = ../Libraries/TestLib.a

SOURCES =	Test.cpp        \
			TestA.cpp       \
			TestB.cpp       \
			TestC.cpp       \
			TestD.cpp       \
			TestE.cpp       \
			TestF.cpp       \
			TestFactory.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(CFLAGS) -o $@ $(INCLUDES) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets ==============================================================

$(OUTPUT) : $(OBJECTS)
	ar r $@ $^
	ranlib $@

depend:
	makedepend -Y $(INCLUDES) $(SOURCES)
	
# DO NOT DELETE

Test.o: ../Includes/OpenNetK/Constants.h ../Common/Constants.h
Test.o: ../Common/TestLib/Test.h ../Import/Includes/KmsLib/ValueVector.h
Test.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
Test.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
Test.o: ../Includes/OpenNet/StatisticsProvider.h
Test.o: ../Includes/OpenNetK/Adapter_Types.h
Test.o: ../Includes/OpenNet/Function_Forward.h ../Includes/OpenNet/Function.h
Test.o: ../Includes/OpenNet/SourceCode.h ../Includes/OpenNet/Kernel_Forward.h
Test.o: ../Includes/OpenNet/Kernel.h ../Includes/OpenNet/PacketGenerator.h
Test.o: ../Includes/OpenNet/System.h ../Common/OpenNet/Adapter_Statistics.h
Test.o: ../Common/OpenNet/PacketGenerator_Statistics.h
Test.o: ../Common/OpenNetK/Adapter_Statistics.h
TestA.o: ../Includes/OpenNetK/Hardware_Statistics.h
TestA.o: ../Common/OpenNetK/Adapter_Statistics.h TestA.h
TestA.o: ../Common/TestLib/Test.h ../Import/Includes/KmsLib/ValueVector.h
TestA.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
TestA.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
TestA.o: ../Includes/OpenNet/StatisticsProvider.h
TestA.o: ../Includes/OpenNetK/Adapter_Types.h
TestA.o: ../Includes/OpenNet/Function_Forward.h
TestA.o: ../Includes/OpenNet/Function.h ../Includes/OpenNet/SourceCode.h
TestA.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
TestA.o: ../Includes/OpenNet/PacketGenerator.h ../Includes/OpenNet/System.h
TestA.o: ../Common/OpenNet/Adapter_Statistics.h
TestA.o: ../Common/OpenNet/PacketGenerator_Statistics.h
