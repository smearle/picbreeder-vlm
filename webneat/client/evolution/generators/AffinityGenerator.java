/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.evolution.generators;

import client.evolution.Generator;

public class AffinityGenerator extends GeneratorChooser {
	public AffinityGenerator(String affinity) {
		addGenerator(new AddNodeWithinAffinity(affinity), 4);
		addGenerator(new AddAcyclicLinkToAffinity(affinity), 6);
		addGenerator(new MutateLinkWeightsWithinAffinity(affinity), 10);
		addGenerator(new MutateActivationWithinAffinity(affinity), 1);
		lock();
	}
}
