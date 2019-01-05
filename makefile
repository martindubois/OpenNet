
# Author     KMS - Martin Dubois, ing.
# Copyright  (C) 2018-2019 KMS. All rights reserved.
# Product    OpenNet
# File       makefile

# ===== Targers =============================================================

all:	Binaries/ONK_Test Binaries/OpenNet.so Binaries/OpenNet_Test Binaries/OpenNet_Tool Libraries/ONK_Lib.a Libraries/TestLib.a ONK_Pro1000/ONK_Pro1000.ko

clean:
	rm -f Binaries/*
	rm -f OpenNet/*.o
	rm -f OpenNet_Test/*.o
	rm -f Libraries/*.a
	rm -f TestLib/*.o

depend:
	cd ONK_Lib; make depend
	cd ONK_Test; make depend
	cd OpenNet; make depend
	cd OpenNet_Test; make depend
	cd OpenNet_Tool; make depend
	cd TestLib; make depend

prep:
	mkdir Binaries
	mkdir Export
	mkdir Libraries

test: Binaries/OpenNet_Test Libraries/OpenNet.so
	Binaries/OpenNet_Test
	
Binaries/ONK_Test: FORCE
	cd ONK_Test; make
	
Binaries/OpenNet.so: FORCE
	cd OpenNet;	make
	
Binaries/OpenNet_Test: Binaries/OpenNet.so FORCE
	cd OpenNet_Test; make
	
Binaries/OpenNet_Tool: Binaries/OpenNet.so Libraries/TestLib.a FORCE
	cd OpenNet_Tool; make
	
Libraries/ONK_Lib.a: FORCE
	cd ONK_Lib; make
    
Libraries/TestLib.a: FORCE
	cd TestLib; make

ONK_Pro1000/ONK_Pro1000.ko: Libraries/ONK_Lib.a FORCE
	cd ONK_Pro1000; make

FORCE:
