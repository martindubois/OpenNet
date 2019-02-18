
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/OSDep.h
/// \brief      Define the OS dependent function the ONK_Lib calls.
/// \todo       Document functions and structure

#pragma once

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef void * ( * OpenNetK_OSDep_AllocateMemory )( unsigned int aSize_byte );
typedef void   ( * OpenNetK_OSDep_FreeMemory     )( void * aMemory );

typedef void * ( * OpenNetK_OSDep_MapBuffer   )( void * aContext, uint64_t * aPA, uint64_t aDA, unsigned int aSize_byte );
typedef void   ( * OpenNetK_OSDep_UnmapBuffer )( void * aContext, void * aBuffer );

typedef void * ( * OpenNetK_OSDep_MapSharedMemory   )( void * aContext, void * aShared_VA, unsigned int aSize_byte );
typedef void   ( * OpenNetK_OSDep_UnmapSharedMemory )( void * aContext );

#ifdef _KMS_LINUX_

    typedef void ( * OpenNetK_OSDep_LockSpinlock   )( void * aLock );
    typedef void ( * OpenNetK_OSDep_UnlockSpinlock )( void * aLock );

#endif

typedef struct
{
    void * mContext;

    OpenNetK_OSDep_AllocateMemory AllocateMemory;
    OpenNetK_OSDep_FreeMemory     FreeMemory    ;

    OpenNetK_OSDep_MapBuffer   MapBuffer  ;
    OpenNetK_OSDep_UnmapBuffer UnmapBuffer;

    OpenNetK_OSDep_MapSharedMemory   MapSharedMemory  ;
    OpenNetK_OSDep_UnmapSharedMemory UnmapSharedMemory;

    // TODO  OpenNetK.OSDep
    //       Normal (Cleanup) - Also use the spinlock functions on Windows

    #ifdef _KMS_LINUX_

        OpenNetK_OSDep_LockSpinlock   LockSpinlock  ;
        OpenNetK_OSDep_UnlockSpinlock UnlockSpinlock;
        
    #endif

}
OpenNetK_OSDep;
