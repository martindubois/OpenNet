
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONL_Lib/Hardware_WDF.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Hardware.h>

#include <OpenNetK/Hardware_WDF.h>

// ===== ONK_Lib ============================================================
#include "OSDep_WDF.h"

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Hardware_WDF * mHardware_WDF;
}
ChildContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(ChildContext, GetChildContext);

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

extern "C"
{
    static EVT_WDF_INTERRUPT_DISABLE InterruptDisable;
    static EVT_WDF_INTERRUPT_DPC     InterruptDpc    ;
    static EVT_WDF_INTERRUPT_ENABLE  InterruptEnable ;
    static EVT_WDF_INTERRUPT_ISR     InterruptIsr    ;
    static EVT_WDF_TIMER             Tick            ;
    static EVT_WDF_WORKITEM          Work            ;
};

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    // NOT TESTED  Driver.LoadUnlock
    //             WdfDmaEnablerCreate fails<br>
    //             WdfCommonBufferCreate fail<br>
    NTSTATUS Hardware_WDF::Init(WDFDEVICE aDevice, Hardware * aHardware)
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

        ASSERT(NULL != aDevice  );
        ASSERT(NULL != aHardware);

        OSDep_Init(&mOSDep, NULL);

        mDevice   = aDevice  ;
        mHardware = aHardware;

        mMemCount = 0;

        WDF_OBJECT_ATTRIBUTES lAttr;

        WDF_OBJECT_ATTRIBUTES_INIT(&lAttr);

        lAttr.ParentObject = aDevice;

        WDFSPINLOCK lSpinLock;

        NTSTATUS lStatus = WdfSpinLockCreate(&lAttr, &lSpinLock);
        ASSERT(STATUS_SUCCESS == lStatus  );
        ASSERT(NULL           != lSpinLock);
        (void)(lStatus);

        mZone0.SetLock (lSpinLock);
        mZone0.SetOSDep(&mOSDep  );

        mHardware->Init(&mZone0);

        NTSTATUS lResult = STATUS_SUCCESS;

        unsigned int lSize_byte = mHardware->GetCommonBufferSize();
        if (0 < lSize_byte)
        {
            WDF_DMA_ENABLER_CONFIG lConfig;

            WDF_DMA_ENABLER_CONFIG_INIT(&lConfig, WdfDmaProfileScatterGather, mHardware->GetPacketSize());

            lResult = WdfDmaEnablerCreate(mDevice, &lConfig, WDF_NO_OBJECT_ATTRIBUTES, &mDmaEnabler);
            if (STATUS_SUCCESS == lResult)
            {
                lResult = WdfCommonBufferCreate(mDmaEnabler, lSize_byte, WDF_NO_OBJECT_ATTRIBUTES, &mCommonBuffer);
                if (STATUS_SUCCESS == lResult)
                {
                    PHYSICAL_ADDRESS lLogicalAddress = WdfCommonBufferGetAlignedLogicalAddress(mCommonBuffer);
                    void           * lVirtualAddress = WdfCommonBufferGetAlignedVirtualAddress(mCommonBuffer);

                    ASSERT(NULL != lVirtualAddress);

                    memset((void *)(lVirtualAddress), 0, lSize_byte); // volatile_cast

                    mHardware->SetCommonBuffer(lLogicalAddress.QuadPart, lVirtualAddress);
                }
            }
        }

        if (STATUS_SUCCESS == lResult)
        {
            InitTimer   ();
            InitWorkItem();
        }

        DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ " - End" DEBUG_EOL);
        return lResult;
    }

    NTSTATUS Hardware_WDF::D0Entry(WDF_POWER_DEVICE_STATE aPreviousState)
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "(  )" DEBUG_EOL);

        ASSERT(NULL != mHardware);
        ASSERT(NULL != mTimer   );

        (void)(aPreviousState);

        mHardware->D0_Entry();

        BOOLEAN lRetB = WdfTimerStart(mTimer, 1000);
        ASSERT(!lRetB);
        (void)(lRetB);

        DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ " - OK" DEBUG_EOL);

        return STATUS_SUCCESS;
    }

    NTSTATUS Hardware_WDF::D0Exit(WDF_POWER_DEVICE_STATE aTargetState)
    {
        ASSERT(NULL != mTimer);

        (void)(aTargetState);

        BOOLEAN lRetB = WdfTimerStop(mTimer, FALSE);
        ASSERT(lRetB);
        (void)(lRetB);

        return mHardware->D0_Exit() ? STATUS_SUCCESS : STATUS_UNSUCCESSFUL;
    }

    // NOT TESTED  ONK_Lib.Hardware_WDF.ErrorHandling
    //             PrepareInterrupt or PrepareMemory fail
    NTSTATUS Hardware_WDF::PrepareHardware(WDFCMRESLIST aRaw, WDFCMRESLIST aTranslated)
    {
        DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ "( ,  )" DEBUG_EOL);

        ASSERT(NULL != aRaw       );
        ASSERT(NULL != aTranslated);

        ULONG lCount = WdfCmResourceListGetCount(aTranslated);

        NTSTATUS lResult = STATUS_SUCCESS;

        for (unsigned int i = 0; i < lCount; i++)
        {
            CM_PARTIAL_RESOURCE_DESCRIPTOR * lDesc = WdfCmResourceListGetDescriptor(aTranslated, i);
            switch (lDesc->Type)
            {
            case CmResourceTypeInterrupt: lResult = PrepareInterrupt(lDesc, WdfCmResourceListGetDescriptor(aRaw, i)); break;
            case CmResourceTypeMemory   : lResult = PrepareMemory   (lDesc); break;
            }

            if (STATUS_SUCCESS != lResult)
            {
                NTSTATUS lRet = ReleaseHardware(aTranslated);
                ASSERT(STATUS_SUCCESS == lRet);

                (void)(lRet);

                break;
            }
        }

        DbgPrintEx(DEBUG_ID, DEBUG_ENTRY_POINT, PREFIX __FUNCTION__ " - End" DEBUG_EOL);

        return lResult;
    }

    NTSTATUS Hardware_WDF::ReleaseHardware(WDFCMRESLIST aTranslated)
    {
        ASSERT(NULL != aTranslated);

        ASSERT(NULL != mHardware);

        (void)(aTranslated);

        mHardware->ResetMemory();

        for (unsigned int i = 0; i < mMemCount; i++)
        {
            ASSERT(NULL != mMem_MA      [i]);
            ASSERT(   0 <  mMemSize_byte[i]);

            MmUnmapIoSpace((PVOID)(mMem_MA[i]), mMemSize_byte[i]); // volatile_cast
        }

        mMemCount = 0;

        return STATUS_SUCCESS;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    NTSTATUS Hardware_WDF::Interrupt_Disable()
    {
        ASSERT(NULL != mHardware);

        mHardware->Interrupt_Disable();

        return STATUS_SUCCESS;
    }

    // CRITICAL PATH  Interrupt
    //                1 / hardware interrupt + 1 / tick
    void Hardware_WDF::Interrupt_Dpc()
    {
        ASSERT(NULL != mHardware);
        ASSERT(NULL != mWorkItem);

        bool lNeedMoreProcessing = false;

        mHardware->Interrupt_Process2(&lNeedMoreProcessing);

        if (lNeedMoreProcessing)
        {
            WdfWorkItemEnqueue(mWorkItem);
        }
    }

    NTSTATUS Hardware_WDF::Interrupt_Enable()
    {
        ASSERT(NULL != mHardware);

        mHardware->Interrupt_Enable();

        return STATUS_SUCCESS;
    }

    // CRITICAL PATH  Interrupt
    //                1 / hardware interrupt
    BOOLEAN Hardware_WDF::Interrupt_Isr(ULONG aMessageId)
    {
        ASSERT(NULL != mHardware );
        ASSERT(NULL != mInterrupt);

        bool lNeedDpc = false;

        BOOLEAN lResult = mHardware->Interrupt_Process(aMessageId, &lNeedDpc) ? TRUE : FALSE;

        if (lNeedDpc)
        {
            TrigProcess2();
        }

        return lResult;
    }

    // CRITICAL PATH  Interrupt
    //                1 / hardware interrupt + 1 / tick
    void Hardware_WDF::TrigProcess2()
    {
        if (NULL != mInterrupt)
        {
            WdfInterruptQueueDpcForIsr(mInterrupt);
        }
    }

    // CRITICAL PATH  Interrupt
    //                1 / tick
    void Hardware_WDF::Tick()
    {
        ASSERT(NULL != mHardware);

        mHardware->Tick();

        TrigProcess2();
    }

    void Hardware_WDF::Work()
    {
        ASSERT(NULL != mHardware);

        mHardware->Interrupt_Process3();
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    void Hardware_WDF::InitTimer()
    {
        ASSERT(NULL != mDevice);

        WDF_TIMER_CONFIG lConfig;

        WDF_TIMER_CONFIG_INIT(&lConfig, ::Tick);

        lConfig.Period = 100;

        WDF_OBJECT_ATTRIBUTES lAttr;

        WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttr, ChildContext);

        lAttr.ParentObject = mDevice;

        NTSTATUS lStatus = WdfTimerCreate(&lConfig, &lAttr, &mTimer);
        ASSERT(STATUS_SUCCESS == lStatus);
        ASSERT(NULL           != mTimer );
        (void)(lStatus);

        ChildContext * lChildContext = GetChildContext(mTimer);
        ASSERT(NULL != lChildContext);

        lChildContext->mHardware_WDF = this;
    }

    void Hardware_WDF::InitWorkItem()
    {
        ASSERT(NULL != mDevice);

        WDF_WORKITEM_CONFIG lConfig;

        WDF_WORKITEM_CONFIG_INIT(&lConfig, ::Work);

        WDF_OBJECT_ATTRIBUTES lAttr;

        WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttr, ChildContext);

        lAttr.ParentObject = mDevice;

        NTSTATUS lStatus = WdfWorkItemCreate(&lConfig, &lAttr, &mWorkItem);
        ASSERT(STATUS_SUCCESS == lStatus  );
        ASSERT(NULL           != mWorkItem);
        (void)(lStatus);

        ChildContext * lChildContext = GetChildContext(mWorkItem);
        ASSERT(NULL != lChildContext);

        lChildContext->mHardware_WDF = this;
    }

    // aTranslated [---;R--]
    // aRaw        [---;R--]
    //
    // Return  STATUS_SUCCESS
    //         See WdfInterruptCreate
    //
    // Thread  PnP
    NTSTATUS Hardware_WDF::PrepareInterrupt(CM_PARTIAL_RESOURCE_DESCRIPTOR * aTranslated, CM_PARTIAL_RESOURCE_DESCRIPTOR * aRaw)
    {
        ASSERT(NULL != aTranslated);
        ASSERT(NULL != aRaw       );

        ASSERT(NULL != mDevice);

        NTSTATUS lResult = STATUS_SUCCESS;

        if (0 == mIntCount)
        {
            WDF_INTERRUPT_CONFIG lConfig;

            WDF_INTERRUPT_CONFIG_INIT(&lConfig, InterruptIsr, InterruptDpc);

            lConfig.EvtInterruptDisable = InterruptDisable;
            lConfig.EvtInterruptEnable  = InterruptEnable ;
            lConfig.InterruptRaw        = aRaw            ;
            lConfig.InterruptTranslated = aTranslated     ;

            WDF_OBJECT_ATTRIBUTES lAttr;

            WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttr, ChildContext);

            lResult = WdfInterruptCreate(mDevice, &lConfig, &lAttr, &mInterrupt);

            ChildContext * lContext = GetChildContext(mInterrupt);
            ASSERT(NULL != lContext);

            lContext->mHardware_WDF = this;
        }

        mIntCount ++;

        return lResult;
    }

    // aTranslated [---;R--]
    //
    // Return  STATUS_OK
    //         STATUS_INSUFFICIENT_RESOURCES
    //         STATUS_UNSUCCESSFUL
    //
    // Thread  PnP

    // NOT TESTED  ONK_Lib.Hardware_WDF.ErrorHandling
    //             MmMapIoSpace fail<br>
    //             Hardware::SetMemory fail
    NTSTATUS Hardware_WDF::PrepareMemory(CM_PARTIAL_RESOURCE_DESCRIPTOR * aTranslated)
    {
        ASSERT(NULL != aTranslated                 );
        ASSERT(   0 <  aTranslated->u.Memory.Length);

        mMemSize_byte[mMemCount] = aTranslated->u.Memory.Length;
        mMem_MA      [mMemCount] = MmMapIoSpace(aTranslated->u.Memory.Start, aTranslated->u.Memory.Length, MmNonCached);

        if (NULL == mMem_MA[mMemCount])
        {
            return STATUS_INSUFFICIENT_RESOURCES;
        }

        if (!mHardware->SetMemory(mMemCount, mMem_MA[mMemCount], mMemSize_byte[mMemCount]))
        {
            return STATUS_UNSUCCESSFUL;
        }

        mMemCount++;

        return STATUS_SUCCESS;
    }

}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

NTSTATUS InterruptDisable(WDFINTERRUPT aInterrupt, WDFDEVICE aAssociatedDevice)
{
    ASSERT(NULL != aInterrupt);

    (void)(aAssociatedDevice);

    ChildContext * lContext = GetChildContext(aInterrupt);
    ASSERT(NULL != lContext               );
    ASSERT(NULL != lContext->mHardware_WDF);

    return lContext->mHardware_WDF->Interrupt_Disable();
}

// CRITICAL PATH  Interrupt
//                1 / hardware interrupt + 1 / tick
VOID InterruptDpc(WDFINTERRUPT aInterrupt, WDFOBJECT aAssociatedObject)
{
    ASSERT(NULL != aInterrupt);

    (void)(aAssociatedObject);

    ChildContext * lContext = GetChildContext(aInterrupt);
    ASSERT(NULL != lContext               );
    ASSERT(NULL != lContext->mHardware_WDF);

    lContext->mHardware_WDF->Interrupt_Dpc();
}

NTSTATUS InterruptEnable(WDFINTERRUPT aInterrupt, WDFDEVICE aAssociatedDevice)
{
    ASSERT(NULL != aInterrupt);

    (void)(aAssociatedDevice);

    ChildContext * lContext = GetChildContext(aInterrupt);
    ASSERT(NULL != lContext               );
    ASSERT(NULL != lContext->mHardware_WDF);

    return lContext->mHardware_WDF->Interrupt_Enable();
}

// CRITICAL PATH  Interrupt
//                1 / hardware interrupt
BOOLEAN InterruptIsr(WDFINTERRUPT aInterrupt, ULONG aMessageId)
{
    ASSERT(NULL != aInterrupt);

    ChildContext * lContext = GetChildContext(aInterrupt);
    ASSERT(NULL != lContext               );
    ASSERT(NULL != lContext->mHardware_WDF);

    return lContext->mHardware_WDF->Interrupt_Isr(aMessageId);
}

VOID Tick(WDFTIMER aTimer)
{
    ASSERT(NULL != aTimer);

    ChildContext * lChildContext = GetChildContext(aTimer);
    ASSERT(NULL != lChildContext);

    lChildContext->mHardware_WDF->Tick();
}

VOID Work(WDFWORKITEM aWorkItem)
{
    ASSERT(NULL != aWorkItem);

    ChildContext * lChildContext = GetChildContext(aWorkItem);
    ASSERT(NULL != lChildContext);

    lChildContext->mHardware_WDF->Work();
}
