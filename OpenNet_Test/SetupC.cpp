
// Author   KMA - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/SetupC.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <memory.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet_Test =======================================================
#include "SetupC.h"

// Public
/////////////////////////////////////////////////////////////////////////////

SetupC::SetupC(unsigned int aBufferQty) : mBufferQty(aBufferQty), mProcessor(NULL), mSystem(NULL)
{
    for (unsigned int i = 0; i < 2; i++)
    {
        mAdapters[i] = NULL;
    }

    memset(&mStats, 0, sizeof(mStats));
}

SetupC::~SetupC()
{
    if (NULL != mSystem)
    {
        mSystem->Delete();
    }
}

int SetupC::Init()
{
    mSystem = OpenNet::System::Create();
    if (NULL == mSystem) { return __LINE__; }

    unsigned int i;

    for (i = 0; i < 2; i++)
    {
        mAdapters[i] = mSystem->Adapter_Get(i);
        if (NULL == mAdapters[i]) { return __LINE__; }
    }

    mProcessor = mSystem->Processor_Get(0);
    if (NULL == mProcessor) { return __LINE__; }

    for (i = 0; i < 2; i++)
    {
        if (OpenNet::STATUS_OK != mSystem->Adapter_Connect(mAdapters[i])) { return __LINE__; }
    }

    for (i = 0; i < 2; i++)
    {
        if (OpenNet::STATUS_OK != mAdapters[i]->SetProcessor(mProcessor)) { return __LINE__; }
    }

    return 0;
}

int SetupC::Start()
{
    unsigned int i;

    for (i = 0; i < 2; i++)
    {
        if (OpenNet::STATUS_OK != mAdapters[i]->SetInputFilter(mFilters + i)) { return __LINE__; }
    }

    for (i = 0; i < 2; i++)
    {
        if (OpenNet::STATUS_OK != mAdapters[i]->Buffer_Allocate(mBufferQty)) { return __LINE__; }
    }

    if (OpenNet::STATUS_OK != mSystem->Start()) { return __LINE__; }

    Sleep(1000);

    return 0;
}

int SetupC::Stop(unsigned int aFlags)
{
    if (OpenNet::STATUS_OK != mSystem->Stop(aFlags)) { return __LINE__; }

    unsigned int i;

    for (i = 0; i < 2; i++)
    {
        if (OpenNet::STATUS_OK != mAdapters[i]->Buffer_Release(mBufferQty)) { return __LINE__; }
    }

    for (i = 0; i < 2; i++)
    {
        if (OpenNet::STATUS_OK != mAdapters[i]->ResetInputFilter()) { return __LINE__; }
    }

    return 0;
}

int SetupC::Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aCount)
{
    for (unsigned int i = 0; i < aCount; i++)
    {
        for (unsigned int j = 0; j < 2; j++)
        {
            if (OpenNet::STATUS_OK != mAdapters[j]->Packet_Send(aPacket, aSize_byte)) { return __LINE__; }
        }
    }

    return 0;
}

int SetupC::Stats_Get()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        if (OpenNet::STATUS_OK != mAdapters[i]->GetStats(mStats + i)) { return __LINE__; }
    }

    return 0;
}

int SetupC::Stats_GetAndDisplay()
{
    if (0 != Stats_Get()) { return __LINE__; }

    for (unsigned int i = 0; i < 2; i++)
    {
        if (OpenNet::STATUS_OK != OpenNet::Adapter::Display(mStats[i], stdout)) { return __LINE__; }
    }

    return 0;
}

int SetupC::Stats_Reset()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        if (OpenNet::STATUS_OK != mAdapters[i]->ResetStats()) { return __LINE__; }
    }

    return 0;
}