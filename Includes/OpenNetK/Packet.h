
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Packet.h
/// \brief      OpenNetK::Packet

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
        /// \brief  Retrieve the physical address of data
        /// \return This method returns a physica address of data
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir l'adresse physique des donnees
        /// \return Cette methode retourne une adresse physique
        /// \endcond
        uint64_t GetData_PA();

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
        void * GetData_XA();

        /// \cond en
        /// \brief  Indicate that the packet now contains received data.
        /// \param  aSize_byte  The size of the received packet
        /// \endcond
        /// \cond fr
        /// \brief  Indiquer que le paquet contient maintenant des donnees
        ///         recu.
        /// \param  aSize_byte  La taille du paquet recu
        /// \endcond
        void IndicateRxCompleted(uint32_t aSize_byte);

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

        uint32_t GetSize();

        void Init(uint64_t aData_PA, void * aData_XA, OpenNet_PacketInfo * aInfo_XA);

    private:

        uint64_t             mData_PA  ;
        void               * mData_XA  ;
        OpenNet_PacketInfo * mInfo_XA  ;
        uint32_t             mSize_byte;

    };

    // Public
    /////////////////////////////////////////////////////////////////////////

    inline uint64_t Packet::GetData_PA()
    {
        return mData_PA;
    }

    inline void * Packet::GetData_XA()
    {
        return mData_XA;
    }

    inline void Packet::IndicateRxCompleted(uint32_t aSize_byte)
    {
        mInfo_XA->mSize_byte = aSize_byte;
        mInfo_XA->mSendTo    =          0;

        mSize_byte = aSize_byte;
        mState     = STATE_RX_COMPLETED;
    }

    inline void Packet::IndicateRxRunning()
    {
        mState = STATE_RX_RUNNING;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    inline uint32_t Packet::GetSize()
    {
        return mSize_byte;
    }

}
