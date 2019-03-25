
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/IoCtl.h
/// \brief      OpenNetK_IoCtl_Info (DDK)

#pragma once

// Data types
/////////////////////////////////////////////////////////////////////////////

/// \cond en
/// \brief  This structure contains information about buffer size an
///         IoCtl accepts.
/// \note   This data type is part of the Driver Development Kit (DDK).
/// \todo   Document members of OpenNetK_IoCtl_Info
/// \endcond
/// \cond fr
/// \brief  Cette structure contient les tailles d'espace m&eacute;moire
///         qu'un IoCtl accepte.
/// \note   Ce type de donn&eacute;e fait partie de l'ensemble de
///         developpement de pilotes (DDK).
/// \todo   Documenter les membre de OpenNetK_IoCtl_Out
/// \endcond
typedef struct
{
    unsigned int mIn_MaxSize_byte ;
    unsigned int mIn_MinSize_byte ;
    unsigned int mOut_MinSize_byte;
}
OpenNetK_IoCtl_Info;

/// \cond en
/// \brief  This enumeration defines the value IoCtl can return.
/// \note   This data type is part of the Driver Development Kit (DDK).
/// \todo   Document values of OpenNetK_IoCtl_Result
/// \endcond
/// \cond fr
/// \brief  Cette enum&eacute;ration d&eacute;finit les valeurs que les IoCtl
///         peuvent retourner.
/// \note   Ce type de donn&eacute;e fait partie de l'ensemble de
///         developpement de pilotes (DDK).
/// \todo   Documenter les valeurs de OpenNetK_IoCtl_Result
/// \endcond
typedef enum
{
    IOCTL_RESULT_OK                = 0x00000000,

    IOCTL_RESULT_PROCESSING_NEEDED = 0xffffffe0,
    IOCTL_RESULT_RETRY             = 0xffffffe1,
    IOCTL_RESULT_WAIT              = 0xffffffe2,

    IOCTL_RESULT_ALREADY_CONNECTED = 0xfffffff1,
    IOCTL_RESULT_CANNOT_DROP       = 0xfffffff2,
    IOCTL_RESULT_CANNOT_MAP_BUFFER = 0xfffffff3,
    IOCTL_RESULT_CANNOT_SEND       = 0xfffffff4,
    IOCTL_RESULT_ERROR             = 0xfffffff5,
    IOCTL_RESULT_INTERRUPTED       = 0xfffffff6,
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
