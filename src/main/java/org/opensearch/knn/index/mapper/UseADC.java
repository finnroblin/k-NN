///*
// * Copyright OpenSearch Contributors
// * SPDX-License-Identifier: Apache-2.0
// */
//
//package org.opensearch.knn.index.mapper;
//
//import lombok.AllArgsConstructor;
//import lombok.Getter;
//
//import java.util.Locale;
//
//@Getter
//@AllArgsConstructor
//public enum UseADC {
//    ADC_ON(true),
//    ADC_OFF(false);
//
//
//    public static UseADC fromName(boolean in) {
//        if (in) {
//            return ADC_ON;
//        }
//        return ADC_OFF;
//        throw new IllegalArgumentException(String.format(Locale.ROOT, "Invalid mode: \"[%s]\"", in));
//    }
//    private static boolean state;
//
//    private static final UseADC DEFAULT = ADC_OFF;
//}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

public enum UseADC {
    ADC_ON(true),
    ADC_OFF(false);

    private final boolean state;

    UseADC(boolean state) {
        this.state = state;
    }

    public boolean getState() {
        return state;
    }

    public static UseADC fromName(boolean in) {
        if (in) {
            return ADC_ON;
        }
        return ADC_OFF;
    }

    public static final UseADC DEFAULT = ADC_OFF;
}
