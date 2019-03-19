
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Connect/Device_Linux.c

// Includes
/////////////////////////////////////////////////////////////////////////////

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    uint32_t          mMsgEnable;
    struct net_device mNetDev   ;

}
DeviceContext;

// Constants
/////////////////////////////////////////////////////////////////////////////

static const struct ethtool_ops ETH_TOOL_OPS =
{
    .get_drvinfo = GetDrvInfo,
};

static const struct net_device_ops NET_DEV_OPS =
{
    .ndo_change_mtu      = ChangeMtu    ,
    .ndo_do_ioctl        = DoIoCtl      ,
    .ndo_features_check  = FeatureCheck ,
    .ndo_fix_features    = FixFeature   ,
    .ndo_get_stats64     = GetStats64   ,
    .ndo_open            = Open         ,
    .ndo_set_features    = SetFeature   ,
    .ndo_set_mac_address = SetMacAddress,
    .ndo_set_rx_mode     = SetRxMode    ,
    .ndo_setup_tc        = SetupTc      ,
    .ndo_start_xmit      = StartXmit    ,
    .ndo_stop            = Stop         ,
    .ndo_tx_timeout      = TxTimeout    ,
    .ndo_validate_addr   = ValidateAddr ,
};

// Functions
/////////////////////////////////////////////////////////////////////////////

void Device_Create()
{
    struct net_device * lNetDev;
    DeviceContext     * lThis  ;
    
    lNetDev = alloc_etherdev_mq(sizeof(DeviceContext), 1);
    if (NULL == lNeDev)
    {
        printf(KERN_ERR "Device_Create - alloc_etherdev_mq( ,  ) failed\n");
        return;
    }

    lThis = netdev_priv(lNetDev);

    lThis->mNetDev = lNetDev;

    lThis->mMsgEnable = netif_msg_init(0, DEFAULT_MSG_ENABLE);

    lThis->mNetDev->ethtool_ops = &ETH_TOOL_OPS;
    lThis->mNetDev->netdev_ops  = &NET_DEV_OPS ;

    strncpy(lThis->mNetDev->name, "ONK_Connect", sizeof(lThis->mNetDev->name) - 1);

}
