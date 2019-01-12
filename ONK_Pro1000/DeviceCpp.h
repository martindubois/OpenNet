
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

extern void DeviceCpp_Init  ( void * aThis );
extern void DeviceCpp_Uninit( void * aThis );

extern unsigned int     DeviceCpp_CommonBuffer_GetSize( void * aThis );
extern void             DeviceCpp_CommonBuffer_Set    ( void * aThis, unsigned long aPhysical, void * aVirtual );

extern ProcessIrqResult DeviceCpp_Interrupt_Process ( void * aThis, unsigned int aMessageId );
extern void             DeviceCpp_Interrupt_Process2( void * aThis );
extern int              DeviceCpp_IoCtl             ( void * aThis, unsigned int aCode, void * aInOut, unsigned int aInSize_byte );
extern int              DeviceCpp_IoCtl_GetInfo     ( unsigned int aCode, unsigned int * aInSizeMax_byte, unsigned int * aInSizeMin_byte, unsigned int * aOutSize_byte );
extern int              DeviceCpp_SetMemory         ( void * aThis, unsigned int aIndex, void * aVirtual, unsigned int aSize_byte );
