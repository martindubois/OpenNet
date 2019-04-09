
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/NVRTCW.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== NVIDIA =============================================================
#include <nvrtc.h>

// Functions
/////////////////////////////////////////////////////////////////////////////

extern void NVRTCW_CompileProgram   ( nvrtcProgram   aProg, int aNumOptions, const char * * aOptions );
extern void NVRTCW_CreateProgram    ( nvrtcProgram * aProg, const char * aSrc, const char * aName, int aNumHeaders, const char * * aHeaders, const char * * aIncludeNames );
extern void NVRTCW_DestroyProgram   ( nvrtcProgram * aProg );
extern void NVRTCW_GetProgramLog    ( nvrtcProgram   aProg, char   * aLog );
extern void NVRTCW_GetProgramLogSize( nvrtcProgram   aProg, size_t * aLogSize_byte );
extern void NVRTCW_GetPTX           ( nvrtcProgram   aProg, char   * aPTX );
extern void NVRTCW_GetPTXSize       ( nvrtcProgram   aProg, size_t * aSize_byte );
