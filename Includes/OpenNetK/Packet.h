
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Packet.h
/// \brief   OpenNetK::Packet

#pragma once

namespace OpenNetK
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class cache packet information
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         method
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe cache les information d'un paquet
    /// \note   Classe noyau - Pas de constructeur, pas de destructor, pas de
    ///         method virtuel
    /// \endcond
    class Packet
    {

    public:

        /// \cond en
        /// \brief  Retrieve the virtual address of data
        /// \return This method returns a virtual address of the kernel
        ///         memory space.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir l'adresse virtuelle des donnees
        /// \return Cette methode retourne une adresse virtuel dans l'espace
        //          d'adressage dy system
        /// \endcond
        void * GetVirtualAddress();

        /// \cond en
        /// \brief  Indicate that the packet now contains received data.
        /// \endcond
        /// \cond fr
        /// \brief  Indiquer que le paquet contient maintenant des donnees
        ///         recu.
        /// \endcond
        void IndicateRxCompleted();

        /// \cond en
        /// \brief  Indicate that the packet is now used to receive data.
        /// \endcond
        /// \cond fr
        /// \brief  Indiquer que le paquet est maintenant utilise pour
        ///         recevoir des donnees.
        /// \endcond
        void IndicateRxRunning();

    // internal:

        // --> TX_RUNNING <-- PX_COMPLETED <--+
        //      |                             |
        //      +--> RX_RUNNING --> RX_COMPLETED
        typedef enum
        {
            STATE_INVALID,

            STATE_PX_COMPLETED,
            STATE_RX_COMPLETED,
            STATE_RX_RUNNING  ,
            STATE_TX_RUNNING  ,

            STATE_QTY
        }
        State;

        uint32_t mSendTo;
        State    mState ;

        uint32_t GetOffset();

        void Init(uint32_t aOffset_byte, void * aVirtualAddress);

    private:

        uint32_t mOffset_byte   ;
        void   * mVirtualAddress;

    };

    // Public
    /////////////////////////////////////////////////////////////////////////

    inline void * Packet::GetVirtualAddress()
    {
        return mVirtualAddress;
    }

    inline void Packet::IndicateRxCompleted()
    {
        mState = STATE_RX_COMPLETED;
    }

    inline void Packet::IndicateRxRunning()
    {
        mState = STATE_RX_RUNNING;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    inline uint32_t Packet::GetOffset()
    {
        return mOffset_byte;
    }

}
