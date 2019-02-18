
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \Copyright  Copyright (C) 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/IoCtl.h
/// \brief      OpenNetK_IoCtl_Info, OpenNetK_IoCtl_Result

#pragma once

// Data types
/////////////////////////////////////////////////////////////////////////////

/// \cond en
/// \brief  This structure contains information about buffer size an
///         IoCtl accepts.
/// \endcond
/// \cond fr
/// \brief  Cette structure contient les tailles d'espace memoire
///         qu'un IoCtl accepte.
/// \endcond
/// \todo   Document the members
typedef struct
{
    unsigned int mIn_MaxSize_byte ;
    unsigned int mIn_MinSize_byte ;
    unsigned int mOut_MinSize_byte;
}
OpenNetK_IoCtl_Info;

// TODO  OpenNetK.IoCtl
//       Normal (Feature) - Ajouter des codes

/// \cond en
/// \brief  This enumeration defines the value IoCtl can return.
/// \endcond
/// \cond fr
/// \brief  Cette enumeration definit les valeurs que les IoCtl peuvent
///         retourner.
/// \endcond
/// \todo   Document the values
typedef enum
{
    IOCTL_RESULT_OK                = 0x00000000,

    IOCTL_RESULT_PROCESSING_NEEDED = 0xffffffe0,

    IOCTL_RESULT_ALREADY_CONNECTED = 0xfffffff5,
    IOCTL_RESULT_ERROR             = 0xfffffff6,
    IOCTL_RESULT_INVALID_PARAMETER = 0xfffffff7,
    IOCTL_RESULT_INVALID_SYSTEM_ID = 0xfffffff8,
    IOCTL_RESULT_NO_BUFFER         = 0xfffffff9,
    IOCTL_RESULT_NOT_SET           = 0xfffffffa,
    IOCTL_RESULT_RUNNING           = 0xfffffffb,
    IOCTL_RESULT_STOPPED           = 0xfffffffc,
    IOCTL_RESULT_SYSTEM_ERROR      = 0xfffffffd,
    IOCTL_RESULT_TOO_MANY_ADAPTER  = 0xfffffffe,
    IOCTL_RESULT_TOO_MANY_BUFFER   = 0xffffffff,
}
OpenNetK_IoCtl_Result;
