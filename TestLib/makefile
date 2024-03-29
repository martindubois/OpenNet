
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       TestLib/makefile

include ../User.mk

INCLUDES = -I ../Includes $(INCLUDE_IMPORT)

OUTPUT = ../Libraries/TestLib.a

SOURCES =	Code.cpp        \
			Test.cpp        \
			TestA.cpp       \
			TestB.cpp       \
			TestC.cpp       \
			TestD.cpp       \
			TestE.cpp       \
			TestF.cpp       \
			TestFactory.cpp

# ===== Rules / Regles ======================================================

.cpp.o:
	g++ -c $(COMPILE_FLAGS) -o $@ $(INCLUDES) $<

# ===== Macros ==============================================================

OBJECTS = $(SOURCES:.cpp=.o)

# ===== Targets ==============================================================

$(OUTPUT) : $(OBJECTS)
	ar r $@ $^
	ranlib $@

depend:
	makedepend -Y $(INCLUDES) $(SOURCES)
	
# DO NOT DELETE

Code.o: Component.h ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Code.o: ../Import/Includes/WindowsToLinux.h Code.h ../Common/TestLib/Code.h
Test.o: Component.h ../Import/Includes/KmsBase.h ../Import/Includes/SafeAPI.h
Test.o: ../Import/Includes/WindowsToLinux.h
Test.o: ../Import/Includes/KmsLib/ThreadBase.h
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
Test.o: ../Common/OpenNetK/Adapter_Statistics.h ../Common/TestLib/Code.h
Test.o: Code.h
TestA.o: Component.h ../Import/Includes/KmsBase.h
TestA.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
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
TestA.o: ../Common/OpenNet/Adapter_Statistics.h ../Common/TestLib/Code.h
TestB.o: Component.h ../Import/Includes/KmsBase.h
TestB.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
TestB.o: ../Includes/OpenNetK/Hardware_Statistics.h TestB.h
TestB.o: ../Common/TestLib/Test.h ../Import/Includes/KmsLib/ValueVector.h
TestB.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
TestB.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
TestB.o: ../Includes/OpenNet/StatisticsProvider.h
TestB.o: ../Includes/OpenNetK/Adapter_Types.h
TestB.o: ../Includes/OpenNet/Function_Forward.h
TestB.o: ../Includes/OpenNet/Function.h ../Includes/OpenNet/SourceCode.h
TestB.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
TestB.o: ../Includes/OpenNet/PacketGenerator.h ../Includes/OpenNet/System.h
TestB.o: ../Common/OpenNet/Adapter_Statistics.h
TestB.o: ../Common/OpenNetK/Adapter_Statistics.h ../Common/TestLib/Code.h
TestC.o: Component.h ../Import/Includes/KmsBase.h
TestC.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
TestC.o: ../Includes/OpenNet/UserBuffer.h ../Includes/OpenNet/Status.h
TestC.o: ../Includes/OpenNet/OpenNet.h
TestC.o: ../Includes/OpenNetK/Hardware_Statistics.h TestC.h
TestC.o: ../Common/TestLib/Test.h ../Import/Includes/KmsLib/ValueVector.h
TestC.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
TestC.o: ../Includes/OpenNet/StatisticsProvider.h
TestC.o: ../Includes/OpenNetK/Adapter_Types.h
TestC.o: ../Includes/OpenNet/Function_Forward.h
TestC.o: ../Includes/OpenNet/Function.h ../Includes/OpenNet/SourceCode.h
TestC.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
TestC.o: ../Includes/OpenNet/PacketGenerator.h ../Includes/OpenNet/System.h
TestC.o: ../Common/OpenNet/Adapter_Statistics.h
TestC.o: ../Common/OpenNetK/Adapter_Statistics.h ../Common/TestLib/Code.h
TestD.o: Component.h ../Import/Includes/KmsBase.h
TestD.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
TestD.o: ../Includes/OpenNetK/Adapter_Types.h
TestD.o: ../Includes/OpenNetK/Hardware_Statistics.h TestD.h
TestD.o: ../Common/TestLib/Test.h ../Import/Includes/KmsLib/ValueVector.h
TestD.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
TestD.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
TestD.o: ../Includes/OpenNet/StatisticsProvider.h
TestD.o: ../Includes/OpenNet/Function_Forward.h
TestD.o: ../Includes/OpenNet/Function.h ../Includes/OpenNet/SourceCode.h
TestD.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
TestD.o: ../Includes/OpenNet/PacketGenerator.h ../Includes/OpenNet/System.h
TestD.o: ../Common/OpenNet/Adapter_Statistics.h
TestD.o: ../Common/OpenNetK/Adapter_Statistics.h ../Common/TestLib/Code.h
TestE.o: Component.h ../Import/Includes/KmsBase.h
TestE.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
TestE.o: ../Includes/OpenNet/Buffer.h ../Includes/OpenNet/Status.h
TestE.o: ../Includes/OpenNet/OpenNet.h
TestE.o: ../Includes/OpenNetK/Hardware_Statistics.h TestE.h
TestE.o: ../Common/TestLib/Test.h ../Import/Includes/KmsLib/ValueVector.h
TestE.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
TestE.o: ../Includes/OpenNet/StatisticsProvider.h
TestE.o: ../Includes/OpenNetK/Adapter_Types.h
TestE.o: ../Includes/OpenNet/Function_Forward.h
TestE.o: ../Includes/OpenNet/Function.h ../Includes/OpenNet/SourceCode.h
TestE.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
TestE.o: ../Includes/OpenNet/PacketGenerator.h ../Includes/OpenNet/System.h
TestE.o: ../Common/OpenNet/Adapter_Statistics.h
TestE.o: ../Common/OpenNetK/Adapter_Statistics.h ../Common/TestLib/Code.h
TestF.o: Component.h ../Import/Includes/KmsBase.h
TestF.o: ../Import/Includes/SafeAPI.h ../Import/Includes/WindowsToLinux.h
TestF.o: ../Includes/OpenNetK/Hardware_Statistics.h TestF.h
TestF.o: ../Common/TestLib/Test.h ../Import/Includes/KmsLib/ValueVector.h
TestF.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
TestF.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
TestF.o: ../Includes/OpenNet/StatisticsProvider.h
TestF.o: ../Includes/OpenNetK/Adapter_Types.h
TestF.o: ../Includes/OpenNet/Function_Forward.h
TestF.o: ../Includes/OpenNet/Function.h ../Includes/OpenNet/SourceCode.h
TestF.o: ../Includes/OpenNet/Kernel_Forward.h ../Includes/OpenNet/Kernel.h
TestF.o: ../Includes/OpenNet/PacketGenerator.h ../Includes/OpenNet/System.h
TestF.o: ../Common/OpenNet/Adapter_Statistics.h
TestF.o: ../Common/OpenNetK/Adapter_Statistics.h ../Common/TestLib/Code.h
TestFactory.o: Component.h ../Import/Includes/KmsBase.h
TestFactory.o: ../Import/Includes/SafeAPI.h
TestFactory.o: ../Import/Includes/WindowsToLinux.h ../Common/Constants.h
TestFactory.o: ../Common/TestLib/TestFactory.h Code.h
TestFactory.o: ../Common/TestLib/Code.h ../Common/TestLib/Test.h
TestFactory.o: ../Import/Includes/KmsLib/ValueVector.h
TestFactory.o: ../Includes/OpenNet/Adapter.h ../Includes/OpenNet/Processor.h
TestFactory.o: ../Includes/OpenNet/OpenNet.h ../Includes/OpenNet/Status.h
TestFactory.o: ../Includes/OpenNet/StatisticsProvider.h
TestFactory.o: ../Includes/OpenNetK/Adapter_Types.h
TestFactory.o: ../Includes/OpenNet/Function_Forward.h
TestFactory.o: ../Includes/OpenNet/Function.h
TestFactory.o: ../Includes/OpenNet/SourceCode.h
TestFactory.o: ../Includes/OpenNet/Kernel_Forward.h
TestFactory.o: ../Includes/OpenNet/Kernel.h
TestFactory.o: ../Includes/OpenNet/PacketGenerator.h
TestFactory.o: ../Includes/OpenNet/System.h
TestFactory.o: ../Common/OpenNet/Adapter_Statistics.h
TestFactory.o: ../Common/OpenNetK/Adapter_Statistics.h TestA.h
TestFactory.o: ../Common/TestLib/Test.h TestB.h TestC.h TestD.h TestE.h
TestFactory.o: TestF.h
