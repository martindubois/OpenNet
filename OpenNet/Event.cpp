
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Event.cpp

#define __CLASS__ "Event::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdint.h>
#include <stdio.h>

// ===== OpenNet ============================================================
#include "Event.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// aProfiling  false  The profiling is disabled
//             true   The profiling is enabled
//
// Exception  KmsLib::Exception *  See CUW_EventCreate
void Event::Init( bool aProfiling )
{
    mProfilling = aProfiling;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// aProfiling  false  The profiling is disabled
//             true   The profiling is enabled
Event::Event( bool aProfiling ) : mExecution_us( 0 ), mProfilling( aProfiling ), mQueued_us( 0 ), mSubmitted_us( 0 )
{
}
