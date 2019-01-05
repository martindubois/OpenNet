
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONL_Lib/Hardware_Linux.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/OS.h>
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Hardware.h>

#include <OpenNetK/Hardware_Linux.h>

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Hardware_Linux * mHardware_Linux;
}
InterruptContext;

typedef struct
{
    OpenNetK::Hardware_Linux * mHardware_Linux;
}
TimerContext;

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

extern "C"
{
};

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    int Hardware_Linux::Init( Hardware * aHardware )
    {
        ASSERT( NULL != aHardware );

        mHardware = aHardware;

        mMemCount = 0;

        new ( & mZone0 ) SpinLock_Linux();

        mHardware->Init( & mZone0 );

        int  lResult = 0;

        unsigned int lSize_byte = mHardware->GetCommonBufferSize();
        if (0 < lSize_byte)
        {
            /* TODO Dev  PHYSICAL_ADDRESS lLogicalAddress = WdfCommonBufferGetAlignedLogicalAddress(mCommonBuffer);
            void           * lVirtualAddress = WdfCommonBufferGetAlignedVirtualAddress(mCommonBuffer);

            ASSERT(NULL != lVirtualAddress);

            memset((void *)(lVirtualAddress), 0, lSize_byte); // volatile_cast

            mHardware->SetCommonBuffer(lLogicalAddress.QuadPart, lVirtualAddress); */

            InitTimer();
        }

        return lResult;
    }

    int Hardware_Linux::D0Entry()
    {
        if ( ! mHardware->D0_Entry() )
        {
            return ( - __LINE__ );
        }

        return 0;
    }

    int Hardware_Linux::D0Exit()
    {
        return mHardware->D0_Exit() ? 0 : ( - __LINE__ );
    }

    int Hardware_Linux::PrepareHardware()
    {
        int lResult = 0;

        return lResult;
    }

    int Hardware_Linux::ReleaseHardware()
    {
        ASSERT(NULL != mHardware);

        mHardware->ResetMemory();

        for (unsigned int i = 0; i < mMemCount; i++)
        {
            ASSERT(NULL != mMemVirtual  [i]);
            ASSERT(   0 <  mMemSize_byte[i]);
        }

        mMemCount = 0;

        return 0;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    int Hardware_Linux::Interrupt_Disable()
    {
        ASSERT( NULL != mHardware );

        mHardware->Interrupt_Disable();

        return 0;
    }

    // CRITICAL PATH
    void Hardware_Linux::Interrupt_Dpc()
    {
        ASSERT(NULL != mHardware);

        mHardware->Interrupt_Process2();
    }

    int Hardware_Linux::Interrupt_Enable()
    {
        ASSERT(NULL != mHardware);

        mHardware->Interrupt_Enable();

        return 0;
    }

    // CRITICAL PATH
    bool Hardware_Linux::Interrupt_Isr( unsigned int aMessageId )
    {
        ASSERT(NULL != mHardware );

        bool lNeedDpc = false;

        bool lResult = mHardware->Interrupt_Process( aMessageId, & lNeedDpc );

        if ( lNeedDpc )
        {
            TrigProcess2();
        }

        return lResult;
    }

    // CRITICAL PATH
    void Hardware_Linux::TrigProcess2()
    {
    }

    void Hardware_Linux::Tick()
    {
        TrigProcess2();
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    void Hardware_Linux::InitTimer()
    {
    }

    // Return  STATUS_OK
    //         STATUS_INSUFFICIENT_RESOURCES
    //         STATUS_UNSUCCESSFUL
    //
    // Thread  PnP

    // NOT TESTED  ONK_Lib.Hardware_WDF.ErrorHandling
    //             MmMapIoSpace fail<br>
    //             Hardware::SetMemory fail
    int Hardware_Linux::PrepareMemory()
    {
        if ( NULL == mMemVirtual[ mMemCount ] )
        {
            return ( - __LINE__ );
        }

        if ( ! mHardware->SetMemory( mMemCount, mMemVirtual[ mMemCount ], mMemSize_byte[ mMemCount ] ) )
        {
            return ( - __LINE__ );
        }

        mMemCount++;

        return 0;
    }

}

// Static functions
/////////////////////////////////////////////////////////////////////////////
