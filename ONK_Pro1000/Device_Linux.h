
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Device_Linux.h

#pragma once

// Functions
/////////////////////////////////////////////////////////////////////////////

extern void * Device_Create( struct pci_dev * aPciDev, unsigned char aMajor, unsigned char aMinor, unsigned int aIndex, struct class * aClass );
extern void   Device_Delete( void * aThis );
