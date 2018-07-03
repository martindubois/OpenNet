
// Author   KMA - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/SetupC.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet_Test =======================================================
#include "SetupC.h"

// Public
/////////////////////////////////////////////////////////////////////////////

SetupC::SetupC(unsigned int aBufferQty) : mBufferQty(aBufferQty), mProcessor(NULL)
{
    for (unsigned int i = 0; i < 2; i++)
    {
        mAdapters[i] = NULL;
    }

    memset(&mStatistics, 0, sizeof(mStatistics));
}

SetupC::~SetupC()
{
}

int SetupC::Init()
{
    assert(NULL == mProcessor);

    if (0 != Base::Init()) { return __LINE__; }

    assert(NULL != mSystem);

    unsigned int i;

    for (i = 0; i < 2; i++)
    {
        assert(NULL == mAdapters[i]);

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
    assert(   0 <  mBufferQty);
    assert(NULL != mSystem   );

    unsigned int i;

    for (i = 0; i < 2; i++)
    {
        OpenNet::Adapter::Config lConfig;

        if (OpenNet::STATUS_OK != mAdapters[i]->GetConfig(&lConfig)) { return __LINE__; }

        lConfig.mBufferQty = mBufferQty;

        if (OpenNet::STATUS_OK != mAdapters[i]->SetConfig( lConfig)) { return __LINE__; }
    }

    for (i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        if (OpenNet::STATUS_OK != mAdapters[i]->SetInputFilter(mFilters + i)) { return __LINE__; }
    }

    if (OpenNet::STATUS_OK != mSystem->Start()) { return __LINE__; }

    Sleep(1000);

    return 0;
}

int SetupC::Stop(unsigned int aFlags)
{
    assert(   0 <  mBufferQty);
    assert(NULL != mSystem   );

    if (OpenNet::STATUS_OK != mSystem->Stop(aFlags)) { return __LINE__; }

    unsigned int i;

    for (i = 0; i < 2; i++)
    {
        if (OpenNet::STATUS_OK != mAdapters[i]->ResetInputFilter()) { return __LINE__; }
    }

    return 0;
}

// aPacket [---;R--]
int SetupC::Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aCount)
{
    for (unsigned int i = 0; i < aCount; i++)
    {
        for (unsigned int j = 0; j < 2; j++)
        {
            assert(NULL != mAdapters[j]);

            if (OpenNet::STATUS_OK != mAdapters[j]->Packet_Send(aPacket, aSize_byte)) { return __LINE__; }
        }
    }

    return 0;
}

int SetupC::Statistics_Get()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        if (OpenNet::STATUS_OK != mAdapters[i]->GetStatistics(mStatistics[i], sizeof(mStatistics[i]), NULL, true)) { return __LINE__; }
    }

    return 0;
}

int SetupC::Statistics_GetAndDisplay(unsigned int aMinLevel)
{
    if (0 != Statistics_Get()) { return __LINE__; }

    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        if (OpenNet::STATUS_OK != mAdapters[i]->DisplayStatistics(mStatistics[i], sizeof(mStatistics[i]), stdout, aMinLevel)) { return __LINE__; }
    }

    return 0;
}

int SetupC::Statistics_Reset()
{
    for (unsigned int i = 0; i < 2; i++)
    {
        assert(NULL != mAdapters[i]);

        if (OpenNet::STATUS_OK != mAdapters[i]->ResetStatistics()) { return __LINE__; }
    }

    return 0;
}

int SetupC::Statistics_Verify(unsigned int aAdapter)
{
    assert(2 > aAdapter);

    assert(NULL != mAdapters[aAdapter]);

    if (0 != KmsLib::ValueVector::Constraint_Verify(mStatistics[aAdapter], STATISTICS_QTY, mConstraints, stdout, reinterpret_cast<const KmsLib::ValueVector::Description *>(mAdapters[aAdapter]->GetStatisticsDescriptions())))
    {
        return __LINE__;
    }

    return 0;
}
