
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/PacketGenerator.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/OpenNet.h>
#include <OpenNet/StatisticsProvider.h>

namespace OpenNet
{

    class Adapter;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  The PacketGenerator class.
    /// \endcond
    /// \cond fr
    /// \brief  La class PacketGenerator.
    /// \endcond
    class PacketGenerator : public StatisticsProvider
    {

    public:

        /// \cond en
        /// \brief  PacketGenerator configuration
        /// \endcond
        /// \cond fr
        /// \brief  Configuration du PacketGenerator
        /// \endcond
        typedef struct
        {
            unsigned int mBandwidth_MiB_s;
            unsigned int mPacketSize_byte;

            unsigned char mReserved0[56];
        }
        Config;

        /// \cond en
        /// \brief   This static methode create an instance of the
        ///          PacketGenerator class.
        /// \return  This static methode return the new instance address.
        /// \endcond
        /// \cond fr
        /// \brief   Cette methode statique cree une instance de la classe
        ///          PacketGenerator.
        /// \return  Cette methode statique retourne l'adresse de la nouvelle
        ///          instance.
        /// \endcond
        OPEN_NET_PUBLIC static PacketGenerator * Create();

        /// \cond en
        /// \brief   This static methode display the PacketGenerator
        //           configuration.
        /// \param   aConfig [---;R--] The configuration
        /// \param   aOut    [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief   Cette methode statique affiche la configuration d'un
        ///          PacketGenerator.
        /// \param   aConfig [---;R--] La configuration
        /// \param   aOut    [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_INVALID_REFERENCE
        /// \retval  STATUS_NOT_ALLOWER_NULL_ARGUMENT
        static OPEN_NET_PUBLIC Status Display(const Config & aConfig, FILE * aOut);

        /// \cond en
        /// \brief   Retrieve the configuration of the PacketGenerator
        /// \param   aOut [---;-W-] The output space
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir la configuration du PacketGenerator
        /// \param   aOut [---;-W-] L'espace memoire de sortie
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status GetConfig(Config * aOut) const = 0;

        /// \cond en
        /// \brief   Connect an Adapter to the PacketGenerator
        /// \param   aAdapter  The Adapter
        /// \endcond
        /// \cond fr
        /// \brief   Connecte un Adapter au PacketGenerator
        /// \param   aAdapter  L'Adapter
        /// \endcond
        /// \retval  STATUS_OK
        virtual Status SetAdapter(Adapter * aAdapter) = 0;

        /// \cond en
        /// \brief   Modify the configuration of the PacketGenerator
        /// \param   aConfig [---;-W-] The configuration
        /// \endcond
        /// \cond fr
        /// \brief   Changer la configuration du PacketGenerator
        /// \param   aConfig [---;-W-] La configuration
        /// \endcond
        /// \retval  STATUS_OK
        /// \retval  STATUS_INVALID_REFERENCE
        virtual Status SetConfig(const Config & aConfig) = 0;

        /// \cond en
        /// \brief   This methode delete the instance.
        /// \endcond
        /// \cond fr
        /// \brief   Cette methode detruit l'instance.
        /// \endcond
        virtual void Delete();

        /// \cond en
        /// \brief  Display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche
        /// \retval aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        virtual Status Display(FILE * aOut) = 0;

        /// \cond en
        /// \brief  Start
        /// \endcond
        /// \cond fr
        /// \brief  Demarrer
        /// \endcond
        /// \retval STATUS_OK
        virtual Status Start() = 0;

        /// \cond en
        /// \brief  Stop
        /// \endcond
        /// \cond fr
        /// \brief  Arreter
        /// \endcond
        /// \retval STATUS_OK
        virtual Status Stop() = 0;

    protected:

        PacketGenerator();

        virtual ~PacketGenerator();

    private:

        PacketGenerator(const PacketGenerator &);

        const PacketGenerator & operator = (const PacketGenerator &);

    };

}
