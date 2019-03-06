
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       Common/Constants.h
//
// This file contains general constants used at user and kernel level.

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

#define SHARED_MEMORY_SIZE_byte (8192)

// ===== Adapter numero =====================================================
#define ADAPTER_NO_QTY     (31)
#define ADAPTER_NO_UNKNOWN (99)

// ===== Packet size ========================================================
#define PACKET_SIZE_MAX_byte (16384)
#define PACKET_SIZE_MIN_byte ( 1536)

// ===== Repeat count =======================================================
// Setting the repeat count too small make impossible for the PacketGenerator
// to send packet as fast it could. At 3072, the PacketGenerator can fill a
// 1 Gb/s link even with 64 bytes packets.
#define REPEAT_COUNT_MAX (3072)
