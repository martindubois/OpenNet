
// Author     KMS - Martin Dubois, P.Eng.
// Copyright  (C) 2019-2020 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Intel_82599.h

// CODE REVIEW 2020-04-14 KMS - Martin Dubois, P.Eng.

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== ONK_Pro1000 ========================================================
#include "Intel.h"
#include "Intel_82599_Regs.h"

namespace Intel_82599
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    class Intel_82599 : public Intel
    {

    public:

        Intel_82599();

        // ===== OpenNetK::Hardware =========================================
        virtual bool SetMemory        (unsigned int aIndex, void * aMemory_MA, unsigned int aSize_byte);
        virtual void D0_Entry         ();
        virtual void Interrupt_Enable ();
        virtual bool Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing);

    protected:

        // ===== Intel ======================================================
        virtual void Config_Apply_Zone0     ();
        virtual void Interrupt_Disable_Zone0();
        virtual void Reset_Zone0            ();
        virtual void Statistics_Update      ();

        // ===== OpenNetK::Hardware =========================================
        virtual void Unlock_AfterReceive_Internal();
        virtual void Unlock_AfterSend_Internal   ();

    private:

        void Rx_Config_Zone0();

        void Tx_Config_Zone0();

        // ===== Zone 0 =====================================================

        volatile BAR1 * mBAR1_82599_MA;

    };

}
