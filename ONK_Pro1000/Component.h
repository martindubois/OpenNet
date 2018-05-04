
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Pro1000/Component.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================

#define INITGUID

#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>

// ===== Includes ===========================================================
#include <OpenNetK/Debug.h>
#include <OpenNetK/StdInt.h>

// Constants
/////////////////////////////////////////////////////////////////////////////

#define DEBUG_ID DPFLTR_IHVDRIVER_ID
#define PREFIX   "ONK_Pro1000: "
