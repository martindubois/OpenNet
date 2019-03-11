
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file    Includes/OpenNetK/Adapter_Types.h
/// \brief   OpenNetK::Adapter_Config, OpenNetK::Adapter_Info,
///          OpenNetK::Adapter_State, OpenNetK::Buffer,
///          OpenNetK::EthernetAddress, OpenNetK::IPv4Address,
///          OpenNetK::VersionInfo

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

namespace OpenNetK
{

    /// \cond en
    /// \brief  This structure is used to pass the information about a buffer.
    /// \note   This data type is part of the Driver Development Kit (DDK).
    /// \todo   Document members of OpenNetK::Buffer
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilis&eacute;e pour passer les
    ///         informations au sujet d'un espace m&eacute;moire.
    /// \todo   Documenter les membres de OpenNetK::Buffer
    /// \endcond
    typedef struct
    {
        uint64_t mBuffer_PA;
        uint64_t mMarker_PA;
        uint32_t mPacketQty;
        uint32_t mSize_byte;

        uint64_t mBuffer_DA;

        uint8_t mReserved1[32];
    }
    Buffer;

    /// \cond en
    /// \brief  This structure is used for Ethernet address.
    /// \note   This data type is part of the Driver Development Kit (DDK).
    /// \todo   Document members of OpenNetK::EthernetAddress
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilis&eacute;e pour les adresses
    ///         Ethernet.
    /// \todo   Documenter les membres de OpenNetL::EthernetAddress
    /// \endcond
    typedef struct
    {
        uint8_t mAddress[6];

        uint8_t mReserved[2];
    }
    EthernetAddress;

    /// \cond en
    /// \brief  This structure is used for IPv4 address.
    /// \note   This data type is part of the Driver Development Kit (DDK).
    /// \todo   Document member of OpenNetK::IPv4Address
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilis&eacute;e pour les adresse IPv4.
    /// \note   Cette classe fait partie de l'ensemble de developpement de
    ///         pilotes (DDK).
    /// \todo   Documenter les membres de OpenNetK::IPv4Address
    /// \endcond
    typedef struct
    {
        uint8_t mAddress[4];
    }
    IPv4Address;

    // TODO OpenNetK.Adapter_Types
    //      High (Feature) - Ajouter des informations : debug/release, date
    //      et heure de compilation, nom du client, type de licence, nombre
    //      de licences

    /// \cond en
    /// \brief  This structure is used to return the version of a component.
    /// \note   This data type is part of the Driver Development Kit (DDK).
    /// \todo   Document members of VersionInfo
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilis&eacute;e pour retourner la version
    ///         d'un composant.
    /// \note   Cette classe fait partie de l'ensemble de developpement de
    ///         pilotes (DDK).
    /// \todo   Documenter les membres de VersionInfo
    /// \endcond
    typedef struct
    {
        uint8_t mMajor        ;
        uint8_t mMinor        ;
        uint8_t mBuild        ;
        uint8_t mCompatibility;

        uint8_t mReserved0[44];

        char mClient     [64];
        char mCompiled_At[32];
        char mCompiled_On[32];
        char mComment    [64];
        char mType       [16];
    }
    VersionInfo;

    /// \cond en
    /// \brief  This enum indicate the adapter type.
    /// \note   This data type is part of the Driver Development Kit (DDK).
    /// \brief  Document values of Adapter_Type
    /// \endcond
    /// \cond fr
    /// \brief  Cette enum&eacute;ration indique le type de l'adaptateur.
    /// \note   Cette classe fait partie de l'ensemble de developpement de
    ///         pilotes (DDK).
    /// \todo   Documenter les valeur de Adapter_Type
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
    /// \note   This data type is part of the Driver Development Kit (DDK).
    /// \todo   Document members of Adapter_Config
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilis&eacute;e pour passer la
    ///         configuration.
    /// \note   Cette classe fait partie de l'ensemble de developpement de
    ///         pilotes (DDK).
    /// \todo   Documenter les membre d'Adapter_Config
    /// \endcond
    typedef struct
    {
        uint32_t mPacketSize_byte;

        uint32_t mReserved0[508];

        EthernetAddress  mEthernetAddress[50];

        uint8_t  mReserved1[112];
    }
    Adapter_Config;

    // TODO  OpenNetK.Adapter
    //       Normal (Feature) - Ajouter la largeur de lien PCIe et la
    //       generation a l'information

    /// \cond en
    /// \brief  This structure is used to return the information about an
    ///         adapter.
    /// \note   This data type is part of the Driver Development Kit (DDK).
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilis&eacute;e pour retourner les
    ///         informations au sujet d'un adaptateur.
    /// \note   Cette classe fait partie de l'ensemble de developpement de
    ///         pilotes (DDK).
    /// \endcond
    /// \todo   Document the members
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

    /// \cond en
    /// \brief  This structure is used to return the status.
    /// \note   This data type is part of the Driver Development Kit (DDK).
    /// \todo   Document members of Adapter_State
    /// \endcond
    /// \cond fr
    /// \brief  Cette structure est utilis&eacute;e pour retourner l'etat.
    /// \note   Cette classe fait partie de l'ensemble de developpement de
    ///         pilotes (DDK).
    /// \todo   Documenter les membres de Adapter_State
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
