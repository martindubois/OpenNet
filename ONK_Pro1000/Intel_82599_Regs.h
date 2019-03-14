
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Intel_82599_Regs.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== ONK_Pro1000 ========================================================
#include "Regs.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define INTEL_82599_BAR1_SIZE (0x124a0)

namespace Intel_82599
{

    // Data types
    /////////////////////////////////////////////////////////////////////////

    // ===== Registers ======================================================

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mFLU               : 1;
            unsigned mANAck2            : 1;
            unsigned mANSF              : 5;
            unsigned m10GPmaPmdParallel : 2;
            unsigned m1GPmaPmd          : 1;
            unsigned mD10GMP            : 1;
            unsigned mRATD              : 1;
            unsigned mRestartAN         : 1;
            unsigned mLMS               : 3;
            unsigned mKRSupport         : 1;
            unsigned mFecR              : 1;
            unsigned mFecA              : 1;
            unsigned mANRxAT            : 4;
            unsigned mANRxDM            : 1;
            unsigned mANRxLM            : 1;
            unsigned mANPDT             : 2;
            unsigned mRF                : 1;
            unsigned mPB                : 2;
            unsigned mKXSupport         : 2;
        }
        mFields;
    }
    AutoC;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 16;

            unsigned m10GPmaPmdSerial : 2;
            unsigned mDDPT            : 1;

            unsigned mReserved1 : 9;

            unsigned mFASM : 1;

            unsigned mReserved2 : 1;

            unsigned mPDD : 1;

            unsigned mReserved3 : 1;
        }
        mFields;
    }
    AutoC2;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mTE : 1;

            unsigned mReserved0 : 2;

            unsigned mGDV : 1;

            unsigned mReserved1 : 12;

            unsigned mVT : 16;
        }
        mFields;
    }
    DmaTxCtl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mRTxQAutoClear : 16;

            unsigned mReserved0 : 14;

            unsigned mTcpTimerAutoClear : 1;

            unsigned mReserved : 1;
        }
        mFields;
    }
    EIAC;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mRTxQ         : 16;
            unsigned mFlowDirector :  1;
            unsigned mRxMiss       :  1;
            unsigned mPCIException :  1;
            unsigned mMailBox      :  1;
            unsigned mLSC          :  1;
            unsigned mLinkSec      :  1;
            unsigned mMng          :  1;

            unsigned mReserved0 : 1;

            unsigned mGPISDP0 : 1;
            unsigned mGPISDP1 : 1;
            unsigned mGPISDP2 : 1;
            unsigned mGDPSDP3 : 1;
            unsigned mECC     : 1;

            unsigned mReserved1 : 1;

            unsigned mTcpTimer            : 1;
            unsigned mOtherCauseInterrupt : 1;
        }
        mFields;
    }
    EI;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 3;

            unsigned mInterval_2us : 9;

            unsigned mReserved1 : 3;

            unsigned mLLIModeration : 1;
            unsigned mLLICredit     : 5;
            unsigned mCounter       : 7;

            unsigned mReserved : 3;

            unsigned mCntWDis : 1;
        }
        mFields;
    }
    EITR;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 3;

            unsigned mTFCE : 2;

            unsigned mReserved1 : 27;
        }
        mFields;
    }
    FCCfg;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 5;

            unsigned mRTH : 14;

            unsigned mReserved1 : 12;

            unsigned mFcEn : 1;
        }
        mFields;
    }
    FCRTH;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 5;

            unsigned mRTL : 14;

            unsigned mReserved1 : 12;

            unsigned mXonE : 1;
        }
        mFields;
    }
    FCRTL;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned int mFCRefreshTh : 16;

            unsigned int mReserved0 : 16;
        }
        mFields;
    }
    FCRTV;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 1;

            unsigned mSBP : 1;

            unsigned mReserved1 : 6;

            unsigned mMPE : 1;
            unsigned mUPE : 1;
            unsigned mBAM : 1;

            unsigned mReserved2 : 21;
        }
        mFields;
    }
    FCtrl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mTTV_0 : 16;
            unsigned mTTV_1 : 16;
        }
        mFields;
    }
    FCTTV;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mSDP0_GPIEn    : 1;
            unsigned mSDP1_GPIEn    : 1;
            unsigned mSDP2_GPIEn    : 1;
            unsigned mSDP3_GPIEn    : 1;
            unsigned mMultipleMSIX  : 1;
            unsigned mOCD           : 1;
            unsigned EIMEn          : 1;
            unsigned LLInterval_4us : 4;
            unsigned RSCDelay_4us   : 3;
            unsigned VT_Mode        : 2;

            unsigned mReserved0 : 14;

            unsigned mEIAME       : 1;
            unsigned mPBA_Support : 1;
        }
        mFields;
    }
    GPIE;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mTxCrcEn    : 1;
            unsigned mRxCrcStrip : 1;
            unsigned mJumboEn    : 1;

            unsigned mReserved1 : 7;

            unsigned mTxPadEn : 1;

            unsigned mReserved2 : 4;

            unsigned mLpbk    : 1;
            unsigned mMdcSpd  : 1;
            unsigned mContMdc : 1;

            unsigned mReserved3 : 2;

            unsigned mPrepend : 4;

            unsigned mReserved4 : 3;

            unsigned mRxLngthErrEn : 1;
            unsigned mRxPadStripEn : 1;

            unsigned mReserved5 : 3;
        }
        mFields;
    }
    HLReg0;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mIntAlloc0 : 6;

            unsigned mReserved0 : 1;

            unsigned mIntAllocVal0 : 1;
            unsigned mIntAlloc1    : 6;

            unsigned mReserved1 : 1;

            unsigned mIntAllocVal1 : 1;
            unsigned mIntAlloc2 : 6;

            unsigned mReserved2 : 1;

            unsigned mIntAllocVal2 : 1;
            unsigned mIntAlloc3    : 6;

            unsigned mReserved3 : 1;

            unsigned mIntAllocVal3 : 1;
        }
        mFields;
    }
    IVAR;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mIntAlloc0    : 7;
            unsigned mIntAllocVal0 : 1;
            unsigned mIntAlloc1    : 7;
            unsigned mIntAllocCal1 : 1;

            unsigned mReserved0 : 16;
        }
        mFields;
    }
    IVAR_Misc;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mKxSigDet          : 1;
            unsigned mFecSigDet         : 1;
            unsigned mFecBlockLock      : 1;
            unsigned mKrHiBErr          : 1;
            unsigned mKrPcsBlockLock    : 1;
            unsigned mKx_ANNPR          : 1;
            unsigned mKx_ANPR           : 1;
            unsigned mLinkStatus        : 1;
            unsigned mKx4SigDet         : 4;
            unsigned mKrSigDet          : 1;
            unsigned m10GLaneSyncStatus : 4;
            unsigned m10GAlignStatus    : 1;
            unsigned m1GSyncStatus      : 1;
            unsigned mK_ANRI            : 1;
            unsigned m1GANEnabled       : 1;
            unsigned m1GLinkEnabled     : 1;
            unsigned m10GLinkEnabled    : 1;
            unsigned mFecEn             : 1;
            unsigned m10GSerEn          : 1;
            unsigned mSGMIIEnabled      : 1;
            unsigned mMLinkMode         : 2;
            unsigned mLinkSpeed         : 2;
            unsigned mLinkUp            : 1;
            unsigned mK_ANC             : 1;
        }
        mFields;
    }
    LinkS;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 16;

            unsigned mMFS_byte : 16;
        }
        mFields;
    }
    MaxFrS;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mPMCF  : 1;
            unsigned mDPF   : 1;
            unsigned mRPFCE : 1;
            unsigned mRFCE  : 1;

            unsigned mReserved0 : 28;
        }
        mFields;
    }
    MFlCn;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 1;

            unsigned mCRCStrip : 1;

            unsigned mReserved1 : 1;

            unsigned mDmaDone : 1;

            unsigned mReserved2 : 13;

            unsigned mRscFrstSize_16byte : 5;

            unsigned mReserved3 : 3;

            unsigned mRscAckC   : 1;
            unsigned mFcoeWrFix : 1;

            unsigned mReserved4 : 5;
        }
        mFields;
    }
    RDRxCtl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mRxEn : 1;

            unsigned mReserved0 : 31;
        }
        mFields;
    }
    RxCtrl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 25;

            unsigned mEnable : 1;

            unsigned mReserved1 : 4;

            unsigned mVME : 1;

            unsigned mReserved2 : 1;
        }
        mFields;
    }
    RxDCtl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mReserved0 : 9;

            unsigned mEccFltEn : 1;

            unsigned mReserved1 : 22;
        }
        mFields;
    }
    RxFecCErr0;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mBSizePacket_KiB : 5;

            unsigned mReserved0 : 3;

            unsigned mBSizeHeader_64bytes : 6;

            unsigned mReserved1 : 8;

            unsigned mRDMTS_64 : 3;
            unsigned mDescType : 3;
            unsigned mDropEn   : 1;

            unsigned mReserved2 : 3;
        }
        mFields;
    }
    SRRCtl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mHead_WB_En : 1;

            unsigned mReserved0 : 1;

            unsigned mHeadWB_Low : 30;
        }
        mFields;
    }
    TDWBAL;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mPThresh : 7;

            unsigned mReserved0 : 1;

            unsigned mHThresh : 7;

            unsigned mReserved1 : 1;

            unsigned mWThresh : 7;

            unsigned mReserved2 : 2;

            unsigned mEnable : 1;
            unsigned mSwFlsh : 1;

            unsigned mReserved3 : 5;
        }
        mFields;
    }
    TxDCtl;

    typedef union
    {
        uint32_t mValue;

        struct
        {
            unsigned mVET : 16;

            unsigned mReserved0 : 12;

            unsigned mCFI   : 1;
            unsigned mCFIEn : 1;
            unsigned mVFE   : 1;

            unsigned mReserved1 : 1;
        }
        mFields;
    }
    VlnCtrl;

    // ===== Groups =========================================================

    typedef struct
    {
        uint32_t mRAL; // 0x00
        uint32_t mRAH; // 0x04
    }
    RxAddress;

    typedef struct
    {
        uint32_t mRDBAL     ; // 0x00
        uint32_t mRDBAH     ; // 0x04
        uint32_t mRDLen_byte; // 0x08

        uint32_t mReserved0;

        uint32_t mRDH   ; // 0x10
        SRRCtl   mSRRCtl; // 0x14
        uint32_t mRDT   ; // 0x18

        uint32_t mReserved1[3];

        RxDCtl   mRxDCtl; // 0x28   0 - 15      16 - 31
        uint32_t mStats0; // 0x30   packet      Dropped_packet
        uint32_t mStats1; // 0x34   Low_byte
        uint32_t mStats2; // 0x38   High_byte

        uint32_t mReserved2[2];
    }
    RxQueue;

    typedef struct
    {
        uint32_t mTDBAL     ; // 0x00
        uint32_t mTDBAH     ; // 0x04
        uint32_t mTDLen_byte; // 0x08

        uint32_t mReserved0;

        uint32_t mTDH; // 0x10

        uint32_t mReserved1;

        uint32_t mTDT; // 0x18

        uint32_t mReserved2[3];

        TxDCtl mTxDCtl; // 0x28

        uint32_t mReserved3[3];

        TDWBAL   mTDWBAL; // 0x38
        uint32_t mTDWBAH; // 0x3c
    }
    TxQueue;

    // ===== BARs ===========================================================

    typedef struct
    {
        REG_RESERVED(00000, 00800);

        EI mEICR_0; // 00800 - Page 572

        REG_RESERVED(00804, 00808);

        EI mEICS_0; // 00808 - Page 573

        REG_RESERVED(0080c, 00810);

        EIAC mEIAC; // 00810 - Page 773

        REG_RESERVED(00814, 00820);

        EITR mEITR_00[24]; // 00820 - Page 575
        EI   mEIMS_0     ; // 00880 - Page 573

        REG_RESERVED(00884, 00888);

        EI mEIMC_0; // 00888 - Page 574

        REG_RESERVED(0088c, 00890);

        EI mEIAM_0; // 00890 - Page 574

        REG_RESERVED(00894, 00898);

        GPIE mGPIE; // 0x00898 - Page 579

        REG_RESERVED(0089c, 00900);

        IVAR      mIVAR [64]; // 00900 - Page 578
        IVAR_Misc mIVAR_Misc; // 00a00 - Page 579

        REG_RESERVED(00a04, 00a90);

        uint32_t mEICS_1[2]; // 00a90 - Page 575

        REG_RESERVED(00a98, 00aa0);

        uint32_t mEIMS_1[2]; // 00aa0 - Page 575

        REG_RESERVED(00aa8, 00ab0);

        uint32_t mEIMC_1[2]; // 00ab0 - Page 575

        REG_RESERVED(00ab8, 00ad0);

        uint32_t mEIAM_1[4]; // 00ad0 - Page 575

        REG_RESERVED(00ae0, 01000);

        RxQueue mRxQueue_00[64]; // 0x1000;

        REG_RESERVED(02000, 02f00);

        RDRxCtl mRDRxCtl; // 02f00 - Page 599

        REG_RESERVED(02f04, 02f50);

        uint32_t mDmaGoodRx_packet  ; // 02f50 - Page 694
        uint32_t mDmaGoodRxLow_byte ; // 02f54 - Page 694
        uint32_t mDmaGoodRxHigh_byte; // 02f58 - Page 694

        REG_RESERVED(02f5c, 03000);

        RxCtrl mRxCtrl; // 03000 - Page 600

        REG_RESERVED(03004, 03200);

        FCTTV mFCTTC[4]; // 03200 - Page 559

        REG_RESERVED(03210, 03220);

        FCRTL mFCRTL[8]; // 03220 - Page 560

        REG_RESERVED(03240, 03260);

        FCRTH mFCRTH[8]; // 03260 - Page 560

        REG_RESERVED(03280, 032a0);

        FCRTV mFCRTV; // 032a0 - Page 561

        REG_RESERVED(032a4, 03c00);

        uint32_t mRxPBSize_byte[8]; // 03c00 - Page 600

        REG_RESERVED(03c20, 03d00);

        FCCfg mFCCfg; // 03d00 - Page 561

        REG_RESERVED(03d04, 04240);

        HLReg0 mHLReg0; // 04240 - Page 666

        REG_RESERVED(04244, 04268);

        MaxFrS mMaxFrS; // 04268 - Page 669

        REG_RESERVED(0426c, 04294);

        MFlCn mMFlCn; // 04294 - Page 685

        REG_RESERVED(04298, 042a0);

        AutoC  mAutoC ; // 042a0 - Page 674
        LinkS  mLinkS ; // 042a4 - Page 676
        AutoC2 mAutoC2; // 042a8 - Page 679

        REG_RESERVED(042ac, 04a80);

        DmaTxCtl mDmaTxCtl; // 04a80 - Page 603

        REG_RESERVED(04a84, 05080);

        FCtrl mFCtrl; // 05080 - Page 582

        REG_RESERVED(05084, 05088);

        VlnCtrl mVlnCtrl; // 05088 - Page 582

        REG_RESERVED(0508c, 051b8);

        RxFecCErr0 mRxFecCErr0; // 051b8 - Page 595

        REG_RESERVED(051bc, 06000);

        TxQueue mTxQueue[128]; // 06000

        REG_RESERVED(08000, 087a0);

        uint32_t mDmaGoodTx_packet  ; // 087a0 - Page 697
        uint32_t mDmaGoodTxLow_byte ; // 087a4 - Page 697
        uint32_t mDmaGoodTxHigh_byte; // 087a8 - Page 697

        REG_RESERVED(087ac, 0a200);

        RxAddress mRxAddress[128]; // 0a200

        REG_RESERVED(0a600, 0d000);

        RxQueue mRxQueue_64[64];

        REG_RESERVED(0e000, 12300);

        EITR mEITR_24[104]; // 12300 - Page 575
    }
    BAR1;

}
