
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Adapter_Types.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

namespace OpenNetK
{

    /// \cond en
    /// \brief  This structure is used to pass the information about a buffer.
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilise pour passer les informations au sujet
    ///         d'un espace memoire.
    /// \endcond
    typedef struct
    {
        uint64_t mBuffer_PA;
        uint64_t mMarker_PA;
        uint32_t mPacketQty;
        uint32_t mSize_byte;

        uint8_t mReserved1[40];
    }
    Buffer;

    /// \cond en
    /// \brief  This structure is used for EthernetAddress.
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilise pour les adresse Ethernet.
    /// \endcond
    typedef struct
    {
        uint8_t mAddress[6];

        uint8_t mReserved[2];
    }
    EthernetAddress;

    /// \cond en
    /// \brief  This structure is used to return the version of a component.
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilise pour retourner la version d'un
    ///         composant.
    /// \endcond
    typedef struct
    {
        uint8_t mMajor        ;
        uint8_t mMinor        ;
        uint8_t mBuild        ;
        uint8_t mCompatibility;

        uint8_t mReserved0[44];

        char mComment[64];
        char mType   [16];
    }
    VersionInfo;

    /// \cond en
    /// \brief  This enum indicate the adapter type.
    /// \endcond
    /// \cond fr
    /// \brief  Cette enum indique le type de l'adaptateur.
    /// \endcond
    typedef enum
    {
        ADAPTER_TYPE_UNKNOWN ,
        ADAPTER_TYPE_ETHERNET,
        ADAPTER_TYPE_NDIS    ,

        ADAPTER_TYPE_QTY
    }
    Adapter_Type;

    /// \cond en
    /// \brief  This structure is used to pass the configuration.
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilise pour passer la configuration.
    /// \endcond
    typedef struct
    {
        uint32_t mPacketSize_byte;

        uint32_t mReserved0[508];

        EthernetAddress  mEthernetAddress[50];

        uint8_t  mReserved1[112];
    }
    Adapter_Config;

    /// \cond en
    /// \brief  This structure is used to return the information about an
    ///         adapter.
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilise pour retourner les information au
    ///         sujet d'un adaptateur.
    /// \endcond
    typedef struct
    {
        Adapter_Type mAdapterType;

        uint32_t mCommonBufferSize_byte;
        uint32_t mPacketSize_byte      ;
        uint32_t mRx_Descriptors       ;
        uint32_t mTx_Descriptors       ;

        uint8_t mReserved0[100];

        EthernetAddress mEthernetAddress;

        char  mComment[128];

        VersionInfo mVersion_Driver  ;
        VersionInfo mVersion_Hardware;
        VersionInfo mVersion_ONK_Lib ;
    }
    Adapter_Info;

    // TODO  OpenNetK.Interface  Add queued buffer state

    /// \cond en
    /// \brief  This structure is used to return the status.
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilise pour retourner l'etat.
    /// \endcond
    typedef struct
    {
        struct
        {
            unsigned mFullDuplex : 1;
            unsigned mLinkUp     : 1;
            unsigned mTx_Off     : 1;

            unsigned mReserved0 : 29;
        }
        mFlags;

        uint32_t mAdapterNo  ;
        uint32_t mBufferCount;
        uint32_t mSpeed_MB_s ;
        uint32_t mSystemId   ;

        uint32_t mReserved0[123];
    }
    Adapter_State;

}