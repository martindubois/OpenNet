
// Author    KMS - Martin Dubois, ing.
// Product   OpenNet
// File      ONL_Lib/Hardware_WDF.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================

#define INITGUID

#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>

// ===== Includes ===========================================================
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Hardware.h>

#include <OpenNetK/Hardware_WDF.h>

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    OpenNetK::Hardware_WDF * mHardware_WDF;
}
InterruptContext;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(InterruptContext, GetInterruptContext)

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry points =======================================================

extern "C"
{
    static EVT_WDF_INTERRUPT_DISABLE InterruptDisable;
    static EVT_WDF_INTERRUPT_DPC     InterruptDpc    ;
    static EVT_WDF_INTERRUPT_ENABLE  InterruptEnable ;
    static EVT_WDF_INTERRUPT_ISR     InterruptIsr    ;
};

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    NTSTATUS Hardware_WDF::Init(WDFDEVICE aDevice, Hardware * aHardware, WDFSPINLOCK aZone0)
    {
        ASSERT(NULL != aDevice  );
        ASSERT(NULL != aHardware);
        ASSERT(NULL != aZone0   );

        mDevice   = aDevice  ;
        mHardware = aHardware;
        mZone0    = aZone0   ;

        mMemCount = 0;

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
                    volatile void  * lVirtualAddress = WdfCommonBufferGetAlignedVirtualAddress(mCommonBuffer);

                    ASSERT(NULL != lVirtualAddress);

                    memset((void *)(lVirtualAddress), 0, lSize_byte); // volatile_cast

                    mHardware->SetCommonBuffer(lLogicalAddress.QuadPart, lVirtualAddress);
                }
            }
        }

        return lResult;
    }

    NTSTATUS Hardware_WDF::D0Entry(WDF_POWER_DEVICE_STATE aPreviousState)
    {
        (void)(aPreviousState);

        return mHardware->D0_Entry() ? STATUS_SUCCESS : STATUS_UNSUCCESSFUL;
    }

    NTSTATUS Hardware_WDF::D0Exit(WDF_POWER_DEVICE_STATE aTargetState)
    {
        (void)(aTargetState);

        return mHardware->D0_Exit() ? STATUS_SUCCESS : STATUS_UNSUCCESSFUL;
    }

    // NOT TESTED  ONK_Lib.Hardware_WDF.ErrorHandling
    //             PrepareInterrupt or PrepareMemory fail
    NTSTATUS Hardware_WDF::PrepareHardware(WDFCMRESLIST aRaw, WDFCMRESLIST aTranslated)
    {
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
            ASSERT(NULL != mMemVirtual  [i]);
            ASSERT(   0 <  mMemSize_byte[i]);

            MmUnmapIoSpace((PVOID)(mMemVirtual[i]), mMemSize_byte[i]); // volatile_cast
        }

        mMemCount = 0;

        return STATUS_SUCCESS;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    NTSTATUS Hardware_WDF::Interrupt_Disable()
    {
        ASSERT(NULL != mHardware);
        ASSERT(NULL != mZone0   );

        WdfSpinLockAcquire(mZone0);
            mHardware->Interrupt_Disable();
        WdfSpinLockRelease(mZone0);

        return STATUS_SUCCESS;
    }

    void Hardware_WDF::Interrupt_Dpc()
    {
        ASSERT(NULL != mHardware);
        ASSERT(NULL != mZone0   );

        WdfSpinLockAcquire(mZone0);
            mHardware->Interrupt_Process2();
        WdfSpinLockRelease(mZone0);
    }

    NTSTATUS Hardware_WDF::Interrupt_Enable()
    {
        ASSERT(NULL != mHardware);
        ASSERT(NULL != mZone0   );

        WdfSpinLockAcquire(mZone0);
            mHardware->Interrupt_Enable();
        WdfSpinLockRelease(mZone0);

        return STATUS_SUCCESS;
    }

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

    void Hardware_WDF::TrigProcess2()
    {
        ASSERT(NULL != mInterrupt);

        WdfInterruptQueueDpcForIsr(mInterrupt);
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

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

            WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&lAttr, InterruptContext);

            lResult = WdfInterruptCreate(mDevice, &lConfig, &lAttr, &mInterrupt);

            InterruptContext * lContext = GetInterruptContext(mInterrupt);
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
        mMemVirtual  [mMemCount] = MmMapIoSpace(aTranslated->u.Memory.Start, aTranslated->u.Memory.Length, MmNonCached);

        if (NULL == mMemVirtual[mMemCount])
        {
            return STATUS_INSUFFICIENT_RESOURCES;
        }

        if (!mHardware->SetMemory(mMemCount, mMemVirtual[mMemCount], mMemSize_byte[mMemCount]))
        {
            return STATUS_UNSUCCESSFUL;
        }

        mMemCount++;

        return STATUS_SUCCESS;
    }

}

// Static functions
/////////////////////////////////////////////////////////////////////////////

NTSTATUS InterruptDisable(WDFINTERRUPT aInterrupt, WDFDEVICE aAssociatedDevice)
{
    ASSERT(NULL != aInterrupt);

    (void)(aAssociatedDevice);

    InterruptContext * lContext = GetInterruptContext(aInterrupt);
    ASSERT(NULL != lContext               );
    ASSERT(NULL != lContext->mHardware_WDF);

    return lContext->mHardware_WDF->Interrupt_Disable();
}

VOID InterruptDpc(WDFINTERRUPT aInterrupt, WDFOBJECT aAssociatedObject)
{
    ASSERT(NULL != aInterrupt);

    (void)(aAssociatedObject);

    InterruptContext * lContext = GetInterruptContext(aInterrupt);
    ASSERT(NULL != lContext               );
    ASSERT(NULL != lContext->mHardware_WDF);

    lContext->mHardware_WDF->Interrupt_Dpc();
}

NTSTATUS InterruptEnable(WDFINTERRUPT aInterrupt, WDFDEVICE aAssociatedDevice)
{
    ASSERT(NULL != aInterrupt);

    (void)(aAssociatedDevice);

    InterruptContext * lContext = GetInterruptContext(aInterrupt);
    ASSERT(NULL != lContext               );
    ASSERT(NULL != lContext->mHardware_WDF);

    return lContext->mHardware_WDF->Interrupt_Enable();
}

BOOLEAN InterruptIsr(WDFINTERRUPT aInterrupt, ULONG aMessageId)
{
    ASSERT(NULL != aInterrupt);

    InterruptContext * lContext = GetInterruptContext(aInterrupt);
    ASSERT(NULL != lContext               );
    ASSERT(NULL != lContext->mHardware_WDF);

    return lContext->mHardware_WDF->Interrupt_Isr(aMessageId);
}