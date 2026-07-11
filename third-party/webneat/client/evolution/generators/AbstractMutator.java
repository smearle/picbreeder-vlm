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

import client.evolution.*;

import java.util.*;
import client.utilities.Random;

/**
 * 
 * 
 */
public abstract class AbstractMutator implements Mutator {
    public final Genome generate(Collection <Genome> parents) {
    	Genome parent = select(parents);
		Genome offspring = GeneticFactoryInstance.get().copyGenome(parent);
		
		mutate(offspring);
		offspring.addParent(parent);
		return offspring;
    }
    
    public final int minimumParents() {
    	return 1;
    }

    /**
     * Selects a parents from the possible parents.
     * 
     * @param parents
     * @return
     */
    protected Genome select(Collection <Genome> parents) {
    	return ((List <Genome>)parents).get(Random.instance().nextInt(parents.size()));
    }

    /**
     * Mutates the offspring produced by selecting and cloning one of the parents
     * in the generator method.
     * 
     * @param offspring The offspring to mutate
     */
    public abstract void mutate(Genome offspring);
}
