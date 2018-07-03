
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Status.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdio.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/OpenNet.h>

namespace OpenNet
{

    // Data type
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  The status codes
    /// \endcond
    /// \cond fr
    /// \brief  Les code de status
    /// \endcond
    typedef enum
    {
        STATUS_OK = 0,

        STATUS_ADAPTER_ALREADY_CONNECTED ,
        STATUS_ADAPTER_NOT_CONNECTED     ,
        STATUS_BUFFER_ALLOCATED          ,
        STATUS_BUFFER_TOO_SMALL          ,
        STATUS_CANNOT_OPEN_INPUT_FILE    ,
        STATUS_CANNOT_READ_INPUT_FILE    ,
        STATUS_CODE_ALREADY_SET          ,
        STATUS_CODE_NOT_SET              ,
        STATUS_CORRUPTED_DRIVER_DATA     ,
        STATUS_DESTINATION_ALREADY_SET   , // 10
        STATUS_DESTINATION_NOT_SET       ,
        STATUS_EMPTY_CODE                ,
        STATUS_EMPTY_INPUT_FILE          ,
        STATUS_ERROR_CLOSING_FILE        ,
        STATUS_ERROR_READING_INPUT_FILE  ,
        STATUS_EXCEPTION                 ,
        STATUS_FILTER_ALREADY_SET        ,
        STATUS_FILTER_NOT_SET            ,
        STATUS_FILTER_SET                ,
        STATUS_INPUT_FILE_TOO_LARGE      , // 20
        STATUS_INTERNAL_ERROR            ,
        STATUS_INVALID_ADAPTER           ,
        STATUS_INVALID_BUFFER_COUNT      ,
        STATUS_INVALID_MODE              ,
        STATUS_INVALID_PACKET_SIZE       ,
        STATUS_INVALID_PROCESSOR         ,
        STATUS_INVALID_REFERENCE         ,
        STATUS_IOCTL_ERROR               ,
        STATUS_NO_ADAPTER_CONNECTED      ,
        STATUS_NO_DESTINATION_SET        , // 30
        STATUS_NOT_ALLOWED_NULL_ARGUMENT ,
        STATUS_OPEN_CL_ERROR             ,
        STATUS_PACKET_TOO_LARGE          ,
        STATUS_PACKET_TOO_SMALL          ,
        STATUS_PROCESSOR_ALREADY_SET     ,
        STATUS_PROCESSOR_NOT_SET         ,
        STATUS_PROFILING_ALREADY_DISABLED,
        STATUS_PROFILING_ALREADY_ENABLED ,
        STATUS_SAME_VALUE                ,
        STATUS_SYSTEM_ALREADY_STARTED    , // 40
        STATUS_SYSTEM_NOT_STARTED        ,
        STATUS_TOO_MANY_BUFFER           ,

        STATUS_QTY
    }
    Status;

    // Function
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This methode returns the description of an status code.
    /// \return The address of a constant
    /// \endcond
    /// \cond fr
    /// \brief  Cette methode retourne la descritption d'un code de status.
    /// \return Un pointeur sur une constantes.
    /// \endcond
    extern OPEN_NET_PUBLIC const char * Status_GetDescription(Status aStatus);

    /// \cond en
    /// \brief  This methode returns the name of an status code.
    /// \return The address of a constant
    /// \endcond
    /// \cond fr
    /// \brief  Cette methode retourne le nom d'un code de status.
    /// \return Un pointeur sur une constantes.
    /// \endcond
    extern OPEN_NET_PUBLIC const char * Status_GetName(Status aStatus);

    /// \cond en
    /// \brief  Display
    /// \param  aStatus        The Status to display
    /// \param  aOut [---;RW-] The output stream
    /// \endcond
    /// \cond fr
    /// \brief  Affiche  Le Status
    /// \param  aStatus        Le Status a afficher
    /// \param  aOut [---;RW-] Le fichier de sortie
    /// \endcond
    /// \retval STATUS_OK
    /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
    extern OPEN_NET_PUBLIC Status Status_Display(Status aStatus, FILE * aOut);

}
