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

package tutorials.generators;

/**
 * Tutorial code that demonstrates how to customize evolution.
 * This mutation scheme mutates the activation function of every
 * node to the default sigmoid implementation.
 * <p>
 * Modify the configuration to use this object if you want to test it!
 * 
 * @author Nick
 */

import client.evolution.Genome;
import client.evolution.Node;
import client.evolution.generators.AbstractMutator;

public final class ForceSigmoidsMutator extends AbstractMutator {
	public void mutate(Genome genome) {
		for(Node n : genome.getNodes())
			if(!n.getType().equals("in"))
				n.setActivation("sigmoid(x)");
	}
}
