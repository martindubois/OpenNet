
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_NDIS/Component.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================
#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>
// #include <netadapter.h>

// ===== Includes ===========================================================
#include <OpenNetK/Debug.h>
#include <OpenNetK/StdInt.h>

// Constants
/////////////////////////////////////////////////////////////////////////////

#define DEBUG_ID DPFLTR_IHVNETWORK_ID
#define PREFIX   "ONK_NDIS: "
#define TAG      'NKNO'
