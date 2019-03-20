
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/Device_Linux.h

#pragma once

// Functions
/////////////////////////////////////////////////////////////////////////////

extern void * Device_Create( unsigned char aMajor, unsigned char aMinor, unsigned int aIndex, struct class * aClass );
extern void   Device_Delete( void * aThis );
