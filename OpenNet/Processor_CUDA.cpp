
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor_CUDA.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== OpenNet ============================================================
#include "Processor_CUDA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aDevice
// aDebugLog [-K-;RW-]
//
// Exception  KmsLib::Exception *  See InitInfo
// Threads  Apps
Processor_CUDA::Processor_CUDA( int aDevice, KmsLib::DebugLog * aDebugLog )
    : Processor_Internal( aDebugLog )
{
    assert( NULL != aDebugLog );

    InitInfo();
}

// ===== Processor_Internal =================================================

Processor_CUDA::~Processor_CUDA()
{
}

// Private
/////////////////////////////////////////////////////////////////////////////

void Processor_CUDA::InitInfo()
{
}
