
// Author   KMA - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Base.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet_Test =======================================================
#include "Base.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define HARDWARE_BASE (OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_QTY)

// Public
/////////////////////////////////////////////////////////////////////////////

Base::Base() : mSystem(NULL)
{
}

Base::~Base()
{
    if (NULL != mSystem)
    {
        mSystem->Delete();
    }
}

int Base::Init()
{
    assert(NULL == mSystem);

    mSystem = OpenNet::System::Create();
    if (NULL == mSystem)
    {
        return __LINE__;
    }

    return 0;
}

void Base::Constraint_Init()
{
    KmsLib::ValueVector::Constraint_Init(mConstraints, STATISTICS_QTY);

    mConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_GET_RESET].mMax = 0xffffffff;

    mConstraints[OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_IOCTL_STATISTICS_RESET].mMax = 0xffffffff;

    mConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_GET_RESET].mMax = 0xffffffff;

    mConstraints[HARDWARE_BASE + OpenNetK::HARDWARE_STATS_STATISTICS_RESET].mMax = 0xffffffff;
}
