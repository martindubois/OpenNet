
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All right reserved.
/// \file       Includes/OpenNet/Status.h

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
    /// \todo   Document values
    /// \endcond
    /// \cond fr
    /// \brief  Les code de status
    /// \todo   Documenter les valeurs
    /// \endcond
    typedef enum
    {
        STATUS_OK = 0,

        STATUS_ADAPTER_ALREADY_CONNECTED ,
        STATUS_ADAPTER_ALREADY_SET       ,
        STATUS_ADAPTER_NOT_CONNECTED     ,
        STATUS_ADAPTER_NOT_SET           ,
        STATUS_ADAPTER_RUNNING           ,
        STATUS_BUFFER_ALLOCATED          , // Not used
        STATUS_BUFFER_TOO_SMALL          ,
        STATUS_CANNOT_OPEN_INPUT_FILE    ,
        STATUS_CANNOT_READ_INPUT_FILE    ,
        STATUS_CODE_ALREADY_SET          , // 10
        STATUS_CODE_NOT_SET              ,
        STATUS_CORRUPTED_DRIVER_DATA     ,
        STATUS_DESTINATION_ALREADY_SET   ,
        STATUS_DESTINATION_NOT_SET       ,
        STATUS_EMPTY_CODE                ,
        STATUS_EMPTY_INPUT_FILE          ,
        STATUS_ERROR_CLOSING_FILE        ,
        STATUS_ERROR_READING_INPUT_FILE  ,
        STATUS_EXCEPTION                 ,
        STATUS_FILTER_ALREADY_SET        , // 20
        STATUS_FILTER_NOT_SET            ,
        STATUS_FILTER_SET                ,
        STATUS_INPUT_FILE_TOO_LARGE      ,
        STATUS_INTERNAL_ERROR            ,
        STATUS_INVALID_ADAPTER           ,
        STATUS_INVALID_ARGUMENT_COUNT    ,
        STATUS_INVALID_BANDWIDTH         ,
        STATUS_INVALID_BUFFER_COUNT      ,
        STATUS_INVALID_BUTTON_INDEX      ,
        STATUS_INVALID_COMMAND_INDEX     , // 30
        STATUS_INVALID_INDEX             ,
        STATUS_INVALID_LINK_SPEED        ,
        STATUS_INVALID_MODE              ,
        STATUS_INVALID_OFFSET            ,
        STATUS_INVALID_PACKET_SIZE       ,
        STATUS_INVALID_PAGE_INDEX        ,
        STATUS_INVALID_PROCESSOR         ,
        STATUS_INVALID_PROTOCOL          ,
        STATUS_INVALID_REFERENCE         ,
        STATUS_INVALID_SIZE              , // 40
        STATUS_IOCTL_ERROR               ,
        STATUS_NAME_TOO_LONG             ,
        STATUS_NAME_TOO_SHORT            ,
        STATUS_NO_ADAPTER_CONNECTED      ,
        STATUS_NO_BUFFER                 ,
        STATUS_NO_DESTINATION_SET        ,
        STATUS_NOT_ADMINISTRATOR         ,
        STATUS_NOT_ALLOWED_NULL_ARGUMENT ,
        STATUS_NOT_IMPLEMENTED           ,
        STATUS_OPEN_CL_ERROR             , // 50
        STATUS_PACKET_GENERATOR_RUNNING  ,
        STATUS_PACKET_GENERATOR_STOPPED  ,
        STATUS_PACKET_TOO_LARGE          ,
        STATUS_PACKET_TOO_SMALL          ,
        STATUS_PROCESSOR_ALREADY_SET     ,
        STATUS_PROCESSOR_NOT_SET         ,
        STATUS_PROFILING_ALREADY_DISABLED,
        STATUS_PROFILING_ALREADY_ENABLED ,
        STATUS_REBOOT_REQUIRED           ,
        STATUS_SAME_VALUE                , // 60
        STATUS_SYSTEM_ALREADY_STARTED    ,
        STATUS_SYSTEM_RUNNING            ,
        STATUS_SYSTEM_NOT_STARTED        ,
        STATUS_THREAD_CREATE_ERROR       ,
        STATUS_THREAD_CLOSE_ERROR        ,
        STATUS_THREAD_STOP_TIMEOUT       ,
        STATUS_THREAD_TERMINATE_ERROR    ,
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
    /// \brief  Cette m&eacute;thode retourne la descritption d'un code de
    ///         status.
    /// \return Un pointeur sur une constantes.
    /// \endcond
    extern OPEN_NET_PUBLIC const char * Status_GetDescription(Status aStatus);

    /// \cond en
    /// \brief  This methode returns the name of an status code.
    /// \return The address of a constant
    /// \endcond
    /// \cond fr
    /// \brief  Cette m&eacute;thode retourne le nom d'un code de status.
    /// \return Un pointeur sur une constantes.
    /// \endcond
    extern OPEN_NET_PUBLIC const char * Status_GetName(Status aStatus);

    /// \cond en
    /// \brief  Display
    /// \param  aStatus  The Status to display
    /// \param  aOut     The output stream
    /// \endcond
    /// \cond fr
    /// \brief  Afficher le Status
    /// \param  aStatus  Le Status &agarave; afficher
    /// \param  aOut     Le fichier de sortie
    /// \endcond
    /// \retval STATUS_OK
    /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
    extern OPEN_NET_PUBLIC Status Status_Display(Status aStatus, FILE * aOut);

}
