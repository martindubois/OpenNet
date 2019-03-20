
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Tunnel_IO/DeviceCpp.h

#pragma once

// Functions
/////////////////////////////////////////////////////////////////////////////

extern unsigned int DeviceCpp_GetContextSize( void );

extern void DeviceCpp_Init( void * aThis, OpenNetK_OSDep * aOSDep, void * aAdapterLock, void * aHardwareLock );

extern void DeviceCpp_D0_Entry( void * aThis );
extern void DeviceCpp_D0_Exit ( void * aThis );

extern int DeviceCpp_IoCtl        ( void * aThis, void * aFileObject, unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte );
extern int DeviceCpp_IoCtl_GetInfo( unsigned int aCode, OpenNetK_IoCtl_Info * aInfo );

extern unsigned int DeviceCpp_Read( void * aThis, void * aBuffer_UA, unsigned int aSize_byte );

extern void DeviceCpp_Release( void * aThis, void * aFile );
extern void DeviceCpp_Tick   ( void * aThis );
