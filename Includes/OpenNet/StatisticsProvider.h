
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/StatisticsProvider.h
/// \brief   OpenNet::StatisticsProvider

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/OpenNet.h>
#include <OpenNet/Status.h>

namespace OpenNet
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  The StatisticsProvider class.
    /// \endcond
    /// \cond fr
    /// \brief  La class StatisticsProvider.
    /// \endcond
    class StatisticsProvider
    {

    public:

        /// \cond en
        /// \brief  The StatisticsDescription structure.
        /// \endcond
        /// \cond fr
        /// \brief  La structure StatisticsDescription.
        /// \endcond
        typedef struct
        {
            const char * mName ;
            const char * mUnit ;
            unsigned int mLevel;
        }
        StatisticsDescription;

        /// \cond en
        /// \brief  This methode return the statistics of the adapter.
        /// \param  aOut       [---;-W-] The methode return the statistics
        ///                              here.
        /// \param  aOutSize_byte        The output buffer size
        /// \param  aInfo_byte [--O;-W-] The returned size
        /// \param  aReset               Reset statistics to 0
        /// \endcond
        /// \cond fr
        /// \brief Cette methode retourne les statistiques de l'adaptateur.
        /// \param  aOut       [---;-W-] La medhode retourne les statistiques
        ///                              ici.
        /// \param  aOutSize_byte        La taille de l'espace memoire
        /// \param  aInfo_byte [--O;-W-] La taille retourne
        /// \param  aReset               Remettre les statistiques a zero
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte = NULL, bool aReset = false) = 0;

        /// \cond en
        /// \brief  This methode returns the number of statistics counter.
        /// \return The number of counters
        /// \endcond
        /// \cond fr
        /// \brief  Cette methode retourne le nombre de compteur statistiques.
        /// \return Le nombre de compteurs
        /// \endcond
        OPEN_NET_PUBLIC unsigned int GetStatisticsQty() const;

        /// \cond en
        /// \brief  This methode returns the statistics counter descriptions.
        /// \return This method return the address of an insternal constant.
        /// \endcond
        /// \cond fr
        /// \brief  Cette methode retourne les descriptions des compteurs de
        ///         statistiques.
        /// \return Cette methode retourne l'adresse d'une constante interne.
        /// \endcond
        const OPEN_NET_PUBLIC StatisticsDescription * GetStatisticsDescriptions() const;

        /// \cond en
        /// \brief  This methode resets the statistics of the adapter.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode reset les statistiques de l'adaptateur.
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_IOCTL_ERROR
        virtual Status ResetStatistics() = 0;

        /// \cond en
        /// \brief  Display
        /// \param  aIn  [---;R--] The statistics to display
        /// \param  aInSize_byte   The statistics size
        /// \param  aOut [---;RW-] The output stream
        /// \param  aMinLevel      0 show all statistics
        /// \endcond
        /// \cond fr
        /// \brief  Afficher
        /// \param  aIn  [---;R--] Les statistiques a afficher
        /// \param  aInSize_byte   La taille des statistiques
        /// \param  aOut [---;RW-] Le fichier de sortie
        /// \param  aMinLevel      0 affiche toutes les statistiques
        /// \endcond
        /// \retval STATUS_OK
        OPEN_NET_PUBLIC Status DisplayStatistics(const unsigned int * aIn, unsigned int aInSize_byte, FILE * aOut, unsigned int aMinLevel = 0);

    protected:

        StatisticsProvider(const StatisticsDescription * aStatisticsDescriptions, unsigned int aStatisticsQty);

    private:

        StatisticsProvider(const StatisticsProvider &);

        const StatisticsProvider & operator = (const StatisticsProvider &);

        const StatisticsDescription * mStatisticsDescriptions;
        unsigned int                  mStatisticsQty         ;

    };

}
