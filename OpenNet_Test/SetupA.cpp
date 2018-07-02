
// Author   KMA - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/SetupA.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet_Test =======================================================
#include "SetupA.h"

// Public
/////////////////////////////////////////////////////////////////////////////

SetupA::SetupA(unsigned int aBufferQty) : mAdapter(NULL), mBufferQty(aBufferQty), mProcessor(NULL)
{
    memset(&mStatistics, 0, sizeof(mStatistics));
}

SetupA::~SetupA()
{
}

int SetupA::Init()
{
    assert(NULL == mAdapter  );
    assert(NULL == mProcessor);
    assert(NULL == mSystem   );

    if (0 != Base::Init())
    {
        return __LINE__;
    }

    mAdapter = mSystem->Adapter_Get(0);
    if (NULL == mAdapter)
    {
        return __LINE__;
    }

    mProcessor = mSystem->Processor_Get(0);
    if (NULL == mProcessor)
    {
        return __LINE__;
    }

    return 0;
}

int SetupA::Start()
{
    assert(NULL != mAdapter);
    assert(NULL != mSystem );

    if (OpenNet::STATUS_OK != mAdapter->SetInputFilter (&mFilter  )) { return __LINE__; }
    if (OpenNet::STATUS_OK != mAdapter->Buffer_Allocate(mBufferQty)) { return __LINE__; }
    if (OpenNet::STATUS_OK != mSystem ->Start          (          )) { return __LINE__; }

    Sleep(1000);

    return 0;
}

int SetupA::Stop(unsigned int aFlags)
{
    assert(NULL != mAdapter);
    assert(NULL != mSystem );

    if (OpenNet::STATUS_OK != mSystem ->Stop            (aFlags    )) { return __LINE__; }
    if (OpenNet::STATUS_OK != mAdapter->Buffer_Release  (mBufferQty)) { return __LINE__; }
    if (OpenNet::STATUS_OK != mAdapter->ResetInputFilter(          )) { return __LINE__; }

    return 0;
}

// aPacket [---;R--]
int SetupA::Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aCount)
{
    assert(NULL != aPacket   );
    assert(   0 <  aSize_byte);
    assert(   0 <  aCount    );

    assert(NULL != mAdapter);

    for (unsigned int i = 0; i < aCount; i++)
    {
        if (OpenNet::STATUS_OK != mAdapter->Packet_Send(aPacket, aSize_byte))
        {
            return __LINE__;
        }
    }

    return 0;
}

int SetupA::Statistics_Get()
{
    assert(NULL != mAdapter);

    if (OpenNet::STATUS_OK != mAdapter->GetStatistics(mStatistics, sizeof(mStatistics), NULL, true))
    {
        return __LINE__;
    }

    return 0;
}

int SetupA::Statistics_GetAndDisplay(unsigned int aMinLevel)
{
    assert(NULL != mAdapter);

    if (0 != Statistics_Get())
    {
        return __LINE__;
    }

    if (OpenNet::STATUS_OK != mAdapter->DisplayStatistics(mStatistics, sizeof(mStatistics), stdout, aMinLevel))
    {
        return __LINE__;
    }

    return 0;
}

int SetupA::Statistics_Reset()
{
    assert(NULL != mAdapter);

    if (OpenNet::STATUS_OK != mAdapter->ResetStatistics())
    {
        return __LINE__;
    }

    return 0;
}

// aConstraints [---;R--]
int SetupA::Statistics_Verify(const KmsLib::ValueVector::Constraint_UInt32 * aConstraints)
{
    assert(NULL != aConstraints);

    assert(NULL != mAdapter);

    if (0 != KmsLib::ValueVector::Constraint_Verify(mStatistics, STATISTICS_QTY, aConstraints, stdout, reinterpret_cast<const KmsLib::ValueVector::Description *>(mAdapter->GetStatisticsDescriptions())))
    {
        return __LINE__;
    }

    return 0;
}
