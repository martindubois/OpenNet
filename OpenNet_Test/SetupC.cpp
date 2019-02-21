
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Test/SetupC.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/ThreadBase.h>

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
    // printf( "%s()\n", __FUNCTION__ );
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

    if (OpenNet::STATUS_OK != mSystem->Adapter_Connect(mAdapters[ 0 ])) { return __LINE__; }
    if (OpenNet::STATUS_OK != mAdapters[0]->SetProcessor(mProcessor)) { return __LINE__; }

    return 0;
}

int SetupC::Start(unsigned int aFlags)
{
    assert(NULL != mAdapters[ 0 ]);
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


    if (OpenNet::STATUS_OK != mAdapters[ 0 ]->SetInputFilter( & mKernel ))
    {
        mKernel.Display( stdout );
        return __LINE__;
    }

    if (OpenNet::STATUS_OK != mSystem->Start(aFlags)) { return __LINE__; }

    KmsLib::ThreadBase::Sleep_ms(1000);

    return 0;
}

int SetupC::Stop()
{
    assert(NULL != mAdapters[0]);
    assert(   0 <  mBufferQty);
    assert(NULL != mSystem   );

    if (OpenNet::STATUS_OK != mSystem->Stop()) { return __LINE__; }

    if (OpenNet::STATUS_OK != mAdapters[ 0 ]->ResetInputFilter()) { return __LINE__; }

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
