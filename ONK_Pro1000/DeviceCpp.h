
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/DeviceCpp.h

#pragma once

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef enum
{
    PIR_IGNORED   ,
    PIR_PROCESSED ,
    PIR_TO_PROCESS,
}
ProcessIrqResult;

// Functions
/////////////////////////////////////////////////////////////////////////////

extern unsigned int DeviceCpp_GetContextSize( void );

extern void DeviceCpp_Init  ( void * aThis, OpenNetK_OSDep * aOSDep, void * aAdapterLock, void * aHardwareLock );
extern void DeviceCpp_Uninit( void * aThis );

extern unsigned int DeviceCpp_CommonBuffer_GetSize( void * aThis );
extern void         DeviceCpp_CommonBuffer_Set    ( void * aThis, uint64_t aPhysical, void * aVirtual );

extern void DeviceCpp_D0_Entry( void * aThis );
extern void DeviceCpp_D0_Exit ( void * aThis );

extern void             DeviceCpp_Interrupt_Enable  ( void * aThis );
extern ProcessIrqResult DeviceCpp_Interrupt_Process ( void * aThis, unsigned int aMessageId );
extern void             DeviceCpp_Interrupt_Process2( void * aThis, bool * aNeedMoreProcessing );
extern void             DeviceCpp_Interrupt_Process3( void * aThis );

extern int DeviceCpp_IoCtl        ( void * aThis, void * aFileObject, unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte );
extern int DeviceCpp_IoCtl_GetInfo( unsigned int aCode, OpenNetK_IoCtl_Info * aInfo );

extern void DeviceCpp_Release    ( void * aThis, void * aFile );
extern void DeviceCpp_ResetMemory( void * aThis );
extern int  DeviceCpp_SetMemory  ( void * aThis, unsigned int aIndex, void * aVirtual, unsigned int aSize_byte );
extern void DeviceCpp_Tick       ( void * aThis );
