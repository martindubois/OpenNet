
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/NVRTCW.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet ============================================================
#include "NVRTCW.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

void NVRTCW_CompileProgram( nvrtcProgram aProg, int aNumOptions, const char * * aOptions )
{
    nvrtcResult lRet = nvrtcCompileProgram( aProg, aNumOptions, aOptions );
    if ( NVRTC_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "nvrtcCompileProgram( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

}

// NVCRT_CreateProgram ==> NVCRT_DestroyProgram
void NVRTCW_CreateProgram( nvrtcProgram * aProg, const char * aSrc, const char * aName, int aNumHeaders, const char * * aHeaders, const char * * aIncludeNames )
{
    assert( NULL != aProg );
    assert( NULL != aSrc  );

    // nvcrtCreateProgram ==> nvcrtDestroyProgram  See NVCRT_DestroyProgram
    nvrtcResult lRet = nvrtcCreateProgram( aProg, aSrc, aName, aNumHeaders, aHeaders, aIncludeNames );
    if ( NVRTC_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "nvrtcCreateProgram( , , , , ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

// NVCRT_CreateProgram ==> NVCRT_DestroyProgram
void NVRTCW_DestroyProgram( nvrtcProgram * aProg )
{
    assert( NULL != aProg );

    // nvcrtCreateProgram ==> nvcrtDestroyProgram  See NVCRT_CreateProgram
    nvrtcResult lRet = nvrtcDestroyProgram( aProg );
    if ( NVRTC_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "nvrtcDestroyProgram(  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void NVRTCW_GetProgramLog( nvrtcProgram aProg, char * aLog )
{
    assert( NULL != aLog );

    nvrtcResult lRet = nvrtcGetProgramLog( aProg , aLog );
    if ( NVRTC_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "nvrtcGetProgramLog( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

extern void NVRTCW_GetProgramLogSize( nvrtcProgram aProg, size_t * aLogSize_byte )
{
    assert( NULL != aLogSize_byte );

    nvrtcResult lRet = nvrtcGetProgramLogSize( aProg , aLogSize_byte );
    if ( NVRTC_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "nvrtcGetProgramLogSize( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( 0 < ( * aLogSize_byte ) );
}

void NVRTCW_GetPTX( nvrtcProgram   aProg, char * aPTX )
{
    assert( NULL != aPTX );

    nvrtcResult lRet = nvrtcGetPTX( aProg , aPTX );
    if ( NVRTC_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "nvrtcGetPTX( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }
}

void NVRTCW_GetPTXSize( nvrtcProgram   aProg, size_t * aSize_byte )
{
    assert( NULL != aSize_byte );

    nvrtcResult lRet = nvrtcGetPTXSize( aProg , aSize_byte );
    if ( NVRTC_SUCCESS != lRet )
    {
        throw new KmsLib::Exception( KmsLib::Exception::CODE_UNKNOWN,
            "nvrtcGetPTXSize( ,  ) failed", NULL, __FILE__, __FUNCTION__, __LINE__, lRet );
    }

    assert( 0 < ( * aSize_byte ) );
}
