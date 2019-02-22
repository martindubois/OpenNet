
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/OSDep.h
/// \brief      Define the OS dependent function the ONK_Lib calls.
/// \todo       Document functions

#pragma once

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef void * ( * OpenNetK_OSDep_AllocateMemory )( unsigned int aSize_byte );
typedef void   ( * OpenNetK_OSDep_FreeMemory     )( void * aMemory );

typedef uint64_t ( * OpenNetK_OSDep_GetTimeStamp )( void );

typedef void   ( * OpenNetK_OSDep_LockSpinlock   )( void * aLock );
typedef void   ( * OpenNetK_OSDep_UnlockSpinlock )( void * aLock );

typedef void * ( * OpenNetK_OSDep_MapBuffer   )( void * aContext, uint64_t * aBuffer_PA, uint64_t aBuffer_DA, unsigned int aSize_byte, uint64_t aMarker_PA, volatile void * * aMarker_MA );
typedef void   ( * OpenNetK_OSDep_UnmapBuffer )( void * aContext, void * aBuffer_MA, unsigned int aSize_byte, volatile void * aMarker_MA );

typedef void * ( * OpenNetK_OSDep_MapSharedMemory   )( void * aContext, void * aShared_UA, unsigned int aSize_byte );
typedef void   ( * OpenNetK_OSDep_UnmapSharedMemory )( void * aContext );

/// \cond en
/// \brief  This structure contains pointer to OS dependant functions
/// \endcond
/// \cond fr
/// \brief  Cette structure contient des pointeurs vers les fonctions qui
///         dependes du systeme d'exploitation.
/// \endcond
/// \todo   Document members
typedef struct
{
    void * mContext;

    OpenNetK_OSDep_AllocateMemory AllocateMemory;
    OpenNetK_OSDep_FreeMemory     FreeMemory    ;

    OpenNetK_OSDep_GetTimeStamp GetTimeStamp;

    OpenNetK_OSDep_LockSpinlock   LockSpinlock  ;
    OpenNetK_OSDep_UnlockSpinlock UnlockSpinlock;

    OpenNetK_OSDep_MapBuffer   MapBuffer  ;
    OpenNetK_OSDep_UnmapBuffer UnmapBuffer;

    OpenNetK_OSDep_MapSharedMemory   MapSharedMemory  ;
    OpenNetK_OSDep_UnmapSharedMemory UnmapSharedMemory;

}
OpenNetK_OSDep;
