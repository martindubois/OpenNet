
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All right reserved.
/// \file       Includes/OpenNetK/Windows.h
/// \brief      Include what is needed on Windows

// Includes
/////////////////////////////////////////////////////////////////////////////

#pragma once

// ===== WDM ================================================================

#define INITGUID

#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>

// Macro
/////////////////////////////////////////////////////////////////////////////

#define SIZE_OF(S) sizeof( S )

#define TRACE_DEBUG DbgPrintEx( DEBUG_ID, DEBUG_DEBUG, PREFIX
#define TRACE_ERROR DbgPrintEx( DEBUG_ID, DEBUG_ERROR, PREFIX

#define TRACE_END )
